from coffea.ml_tools.torch_wrapper import torch_wrapper
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define, field
import awkward as ak
import json
import numpy as np
import onnxruntime
import correctionlib
from coffea.nanoevents.methods import vector as coffea_vector

# Restated from abcdiscotec/data/preprocessing.py so the analyzer does not import the training package.
_TRANSFORMS = {"none": lambda x: x, "asinh": np.arcsinh, "log1p": np.log1p}
_SCALERS = ("standard", "robust", "none")


def _read_preprocessing(path: str) -> dict:
    """Load a run's ``preprocessing.json`` and check it is internally consistent:
    known transform and scaler names, and ``loc``/``scale`` of length 1 (shared
    across jets, and always for a global variable) or ``n_jets`` (one per jet
    slot) according to ``shared_scaling_across_jets``."""
    with open(path) as f:
        pre = json.load(f)
    if pre["scaler"] not in _SCALERS:
        raise ValueError(
            f"Unknown scaler {pre['scaler']!r} in {path}; known: {list(_SCALERS)}"
        )
    for name, state in pre["states"].items():
        if state["transform"] not in _TRANSFORMS:
            raise ValueError(
                f"Unknown transform {state['transform']!r} for {name} in {path}; known: {list(_TRANSFORMS)}"
            )
        per_slot = (
            name in pre["jet_variables"] and not pre["shared_scaling_across_jets"]
        )
        expected = pre["n_jets"] if per_slot else 1
        if len(state["loc"]) != expected or len(state["scale"]) != expected:
            raise ValueError(
                f"{name} in {path} has {len(state['loc'])} loc and {len(state['scale'])} scale "
                f"entries, expected {expected}"
            )
    return pre


def _valid_mask(raw: np.ndarray, pad_value) -> np.ndarray:
    """True where the value is not the pad sentinel; all True when none is configured."""
    if pad_value is None:
        return np.ones(raw.shape, dtype=bool)
    if pad_value != pad_value:  # NaN sentinel
        return ~np.isnan(raw)
    return raw != pad_value


def _jet_block(pre: dict, jets: dict) -> tuple:
    """The scaled jet block, shape ``(n_events, n_jets, n_jet_variables)``, and
    its per-jet validity, shape ``(n_events, n_jets)``.

    ``jets`` maps each training jet variable to its jagged per-event array.
    Each is padded to ``n_jets`` with the pad sentinel (0 when training used
    none), transformed, scaled, and set to 0 where it is a pad, which after
    standardization is the feature mean. The arithmetic follows
    ``Preprocessor.transform_jet_block`` step by step, including the dtypes, so
    the values are identical to the ones the model was trained on.
    """
    n_jets, pad_value = pre["n_jets"], pre["pad_value"]
    fill = 0.0 if pad_value is None else pad_value
    n = len(next(iter(jets.values())))
    block = np.empty((n, n_jets, len(pre["jet_variables"])), dtype=np.float32)
    any_invalid = np.zeros((n, n_jets), dtype=bool)

    for j, var in enumerate(pre["jet_variables"]):
        state = pre["states"][var]
        raw = ak.to_numpy(
            ak.fill_none(ak.pad_none(jets[var], n_jets, clip=True), np.float32(fill))
        )
        mask = _valid_mask(raw, pad_value)
        transformed = _TRANSFORMS[state["transform"]](raw)
        loc, scale = np.asarray(state["loc"]), np.asarray(state["scale"])
        if pre["shared_scaling_across_jets"]:
            normalized = (transformed - loc[0]) / scale[0]
        else:
            normalized = (transformed - loc[np.newaxis, :]) / scale[np.newaxis, :]
        block[:, :, j] = np.where(mask, normalized, 0.0).astype(np.float32)
        any_invalid |= ~mask
    return block, ~any_invalid


def _flat_global(arr) -> np.ndarray:
    """One value per event. A column holding 0 or 1 objects per event (a selected
    candidate) is flattened, NaN where the list is empty, as ``SaveEventsH5``
    wrote it."""
    if arr.ndim == 2:
        most = ak.max(ak.num(arr, axis=1))
        if most is not None and most > 1:
            raise ValueError(
                f"A global column has up to {most} entries per event; only 0 or 1 can be flattened."
            )
        arr = ak.fill_none(ak.firsts(arr), np.nan)
    return ak.to_numpy(arr)


def _global_block(pre: dict, globals_: dict) -> np.ndarray:
    """The scaled global variables the model takes, one column each, in the order
    of ``globals_``, following ``Preprocessor.transform_global``."""
    pad_value = pre["pad_value"]
    out = np.empty(
        (len(next(iter(globals_.values()))), len(globals_)), dtype=np.float32
    )
    for j, (var, raw) in enumerate(globals_.items()):
        state = pre["states"][var]
        mask = _valid_mask(raw, pad_value)
        transformed = _TRANSFORMS[state["transform"]](raw)
        normalized = (transformed - state["loc"][0]) / state["scale"][0]
        out[:, j] = np.where(mask, normalized, 0.0).astype(np.float32)
    return out


def _build_features(pre: dict, jets: dict, globals_: dict) -> np.ndarray:
    """The model input, shape ``(n_events, n_features)``, float32.

    The layout is the one ``_process_batch_arrays`` builds for training: every
    jet variable of jet 0, then of jet 1, and so on; then one validity flag per
    jet, only if the run has a pad sentinel; then the global variables.

    Parameters
    ----------
    pre : dict
        The run's ``preprocessing.json``, as read by ``_read_preprocessing``.
    jets : dict[str, ak.Array]
        Training jet variable name -> jagged per-event array.
    globals_ : dict[str, np.ndarray]
        Training name of each global variable the model takes -> one value per event.
    """
    block, valid = _jet_block(pre, jets)
    n = block.shape[0]
    parts = [block.reshape(n, pre["n_jets"] * len(pre["jet_variables"]))]
    if pre["pad_value"] is not None:
        parts.append(valid.astype(np.float32))
    parts.append(_global_block(pre, globals_))
    return np.concatenate(parts, axis=1)


@define
class ABCDiHiggsInference(AnalyzerModule):
    """
    Inference module for a network trained with abcdiscotec.
    This module takes in jet-like columns and global event features,
    prepares the inputs for the neural network exactly as the training did,
    performs inference, and adds the output probabilities to the columns.

    The training run supplies the two files the inputs depend on: the traced
    model (``model_best.pt`` or ``model_last.pt``) and ``preprocessing.json``
    (per-variable transforms and scalings, the jet count, the pad sentinel and
    the global variables left out of the input). The variables named here are
    checked against that file on every call, so a list that does not match
    what the model was trained on raises instead of feeding it mis-scaled
    inputs. Jet variables are matched as ``<jet_col>_<var>`` and global columns
    by their name with ``.`` replaced by ``_``, the rule ``SaveEventsH5`` writes
    the training variables with.

    A matching configuration for the ``withHLT_ObjDef9`` skim::

        - module_name: PromoteAttributeToBracket
          input_col: lowJet
          nest_col: storedE
          attribute_name: e
        - module_name: ABCDiHiggsInference
          jet_col: lowJet
          jet_vars: [pt, eta, phi, storedE, btag, btagPNetQvG, btagUParTAK4CvB, btagUParTAK4CvL]
          global_cols: [HT, nontop2b_qvg_threejet.mass, b_dijet.mass]
          output_cols: [Disc1]
          model_path: nn_models/<run>/model_best.pt
          preprocessing_path: nn_models/<run>/preprocessing.json

    Parameters
    ----------
    jet_col : Column
        Column containing the jet collection.
    jet_vars : list[str]
        Fields of the jet collection the network takes, in the training order.
        ``btag`` resolves to the era's tagger.
    global_cols : list[Column]
        Global event feature columns the network takes, in the training order
        (those not in the run's ``exclude_inputs``).
    model_path : str
        Path to the traced model of the run.
    preprocessing_path : str
        Path to the run's ``preprocessing.json``.
    output_cols : list[Column]
        One column per network head. Each gets one sub-column per class.
    class_names : list[str], optional
        Names of the sub-columns, in the model's output order. Default
        ``[sig, qcd, tt]``.
    """

    jet_col: Column
    jet_vars: list[str]
    global_cols: list[Column]
    model_path: str
    preprocessing_path: str
    output_cols: list[Column]
    class_names: list[str] = field(factory=lambda: ["sig", "qcd", "tt"])

    def _preprocessing(self) -> dict:
        """The run's preprocessing, checked against the variables configured here."""
        pre = _read_preprocessing(self.preprocessing_path)
        jet_prefix = str(self.jet_col).replace(".", "_")
        jet_names = [f"{jet_prefix}_{v}" for v in self.jet_vars]
        if jet_names != pre["jet_variables"]:
            raise ValueError(
                f"jet_col and jet_vars give the variables {jet_names} but the run was trained on "
                f"{pre['jet_variables']}"
            )
        wanted = [v for v in pre["global_variables"] if v not in pre["exclude_inputs"]]
        global_names = [str(c).replace(".", "_") for c in self.global_cols]
        if global_names != wanted:
            raise ValueError(
                f"global_cols give the variables {global_names} but the run's model takes {wanted} "
                f"(its global variables {pre['global_variables']}, excluded from the input: "
                f"{pre['exclude_inputs']})"
            )
        return pre

    def _jet_column(self, columns, name: str):
        """One field of the jet collection, as the (lazy) column; ``btag`` is the era's tagger."""
        field_name = (
            name
            if (name != "btag")
            else columns.metadata["era"]["btag_scale_factors"]["tagger"]
        )
        return columns[self.jet_col + Column(field_name)]

    def prepare_inputs(self, columns, pre=None, first_jet_column=None):
        """The model input. ``first_jet_column`` is the column of ``jet_vars[0]``
        if the caller has already requested it."""
        pre = pre or self._preprocessing()
        jets = {
            var: (
                first_jet_column
                if (i == 0 and first_jet_column is not None)
                else self._jet_column(columns, name)
            )
            for i, (name, var) in enumerate(zip(self.jet_vars, pre["jet_variables"]))
        }
        globals_ = {
            var: _flat_global(columns[col])
            for var, col in zip(
                (str(c).replace(".", "_") for c in self.global_cols), self.global_cols
            )
        }
        return _build_features(pre, jets, globals_)

    def run(self, columns, params):
        n_discs = len(self.output_cols)
        n_classes = len(self.class_names)

        class ABCDiHiggsNetwork(torch_wrapper):
            def prepare_awkward(self, inputs):
                return [
                    ak.values_astype(inputs, "float32"),
                ], {}

            def postprocess_awkward(self, output, events):
                values = ak.to_numpy(output)
                if values.ndim != 3 or values.shape[1:] != (n_discs, n_classes):
                    raise ValueError(
                        f"The model returned an output of shape {values.shape}; expected "
                        f"(n_events, {n_discs} output_cols, {n_classes} class_names)"
                    )
                return {f"Disc{i}": values[:, i] for i in range(n_discs)}

        pre = self._preprocessing()
        first_jet_column = self._jet_column(columns, self.jet_vars[0])
        if len(first_jet_column) == 0:
            empty = np.array([], dtype="float32")
            for col in self.output_cols:
                for name in self.class_names:
                    columns[col + Column(name)] = ak.Array(empty)
            return columns, []

        model = ABCDiHiggsNetwork(self.model_path)
        X = self.prepare_inputs(columns, pre, first_jet_column)
        outputs = model(ak.Array(X))
        for i, col in enumerate(self.output_cols):
            for k, name in enumerate(self.class_names):
                columns[col + Column(name)] = ak.Array(outputs[f"Disc{i}"][:, k])
        return columns, []

    def neededResources(self, metadata):
        return [self.model_path, self.preprocessing_path]

    def outputs(self, metadata):
        return self.output_cols

    def inputs(self, metadata):
        return [self.jet_col] + self.global_cols


# ------------------------------------------ #
# SPANet-related models for SPANet inference #
# ------------------------------------------ #


def _pad_and_convert(arr, n_real_jets, n_null_jets):
    padded = ak.pad_none(arr[:, :n_real_jets], n_real_jets + n_null_jets, clip=True)
    padded = ak.fill_none(padded, 0)
    return ak.to_numpy(padded)


def _make_mask(jet_counts, n_real_jets, n_null_jets):
    return np.array(
        [
            [True] * min(n, n_real_jets)
            + [False] * max(0, n_real_jets - n)
            + [True] * n_null_jets
            for n in jet_counts
        ],
        dtype=bool,
    )


def _order_fields_btag_then_secondary(field_arrays, secondary_key):
    """
    Sort jets: btag descending for the first 2 slots (H1/bb candidates),
    then field_arrays[secondary_key] descending for the remaining slots
    (H2/WW candidates). secondary_key selects which ALREADY-FETCHED field
    ("qvg" or "pt") determines the order -- it does not change which
    fields get fetched or fed to the model as features. This matters:
    a variant training convention may reorder jets by pt instead of QvG
    while still using the genuine QvG score as its own separate input
    feature -- conflating "what determines the order" with "what gets stacked as a
    feature" would silently feed the wrong numbers into the qvg feature
    slot. Every entry in field_arrays, including whichever one ISN'T the
    secondary_key, still gets consistently reordered and still gets fed
    to the model as its own real, untouched feature.
    """
    btag_sort_idx = ak.argsort(field_arrays["btag"], axis=1, ascending=False)
    partially_sorted = {k: v[btag_sort_idx] for k, v in field_arrays.items()}

    secondary_indices = (
        ak.argsort(partially_sorted[secondary_key][:, 2:], axis=1, ascending=False) + 2
    )
    return {
        k: ak.concatenate([v[:, :2], v[secondary_indices]], axis=1)
        for k, v in partially_sorted.items()
    }


def _build_source_data(pt, eta, phi, e, btag, qvg, extra_features=None):
    """Exact transform the model expects: pt/e log1p'd, btag/qvg/ctag fed
    as-is (no log transform -- they're already O(1) scores or labels, not
    physical quantities benefiting from log-compression the way pt/e do).

    extra_features, if given, is a list of arrays appended AFTER qvg, in
    the SAME order they were written into the training Source array by
    skimming.py -- for c-tagging: one array (ctag) in label mode, two
    arrays (ctag_cvb, ctag_cvl) in score mode, matching
    source_fields.update(ctag_source_fields)'s insertion order exactly.
    Column count varies with mode (7 vs 8 total features) -- this MUST
    match whichever mode the specific .onnx model was actually trained
    with, or the stacked array will silently be fed to the wrong input
    slots.
    """
    base = [np.log(pt + 1), eta, phi, np.log(e + 1), btag, qvg]
    if extra_features:
        base.extend(extra_features)
    return np.stack(base, axis=-1).astype(np.float32)


def _run_onnx_inference(
    session,
    source_data,
    mask,
    particle_names,
    batch_size=256,
    assign_suffix="_assignment_probability",
    detect_suffix="_detection_probability",
    extra_output_names=None,
):
    """
    Batched ONNX call. Looks up each particle's assignment/detection
    output BY NAME (via session.get_outputs()), not by fixed position --
    more robust than a hardcoded positional order, and self-documenting
    if a name doesn't match what's expected.

    Returns: dict mapping particle_name -> (assign_array, detect_array).

    If extra_output_names is given (e.g. the classification output), those outputs are also
    collected, and the return value becomes the tuple (results, extras) with extras mapping
    each requested output name to its concatenated array. Without it the return value is
    unchanged.
    """
    output_names = [o.name for o in session.get_outputs()]
    expected_names = [f"{name}{assign_suffix}" for name in particle_names] + [
        f"{name}{detect_suffix}" for name in particle_names
    ]
    missing = [n for n in expected_names if n not in output_names]
    if missing:
        raise KeyError(
            f"Expected ONNX output names not found: {missing}. "
            f"Actual model output names: {output_names}. "
            f"The assumed '<Name>{assign_suffix}'/'<Name>{detect_suffix}' naming "
            f"convention may not match this model -- check session.get_outputs() directly."
        )

    n = len(source_data)
    collected = {name: [] for name in output_names}
    for i in range(0, n, batch_size):
        out = session.run(
            output_names,
            {
                "Source_data": source_data[i : i + batch_size],
                "Source_mask": mask[i : i + batch_size],
            },
        )
        for name, arr in zip(output_names, out):
            collected[name].append(arr)

    results = {}
    for name in particle_names:
        assign = np.concatenate(collected[f"{name}{assign_suffix}"], axis=0)
        detect = np.concatenate(collected[f"{name}{detect_suffix}"], axis=0)
        results[name] = (assign, detect)
    if extra_output_names:
        absent = [n for n in extra_output_names if n not in output_names]
        if absent:
            raise KeyError(
                f"Requested ONNX outputs not found: {absent}. Actual outputs: {output_names}"
            )
        extras = {n: np.concatenate(collected[n], axis=0) for n in extra_output_names}
        return results, extras
    return results


def _find_classification_output(
    output_names,
    particle_names,
    hint=None,
    assign_suffix="_assignment_probability",
    detect_suffix="_detection_probability",
):
    """
    Name of the ONNX output that holds the event-classification scores.

    hint (the module's classification_output setting) wins if given. Otherwise the output is
    inferred: drop the assignment / detection outputs, drop anything that looks like a
    regression output, and take what is left; if several remain, only one containing
    'classif' is accepted. Anything ambiguous raises with the full list of output names, so a
    wrong guess can never silently feed the wrong tensor into the score columns.
    """
    if hint:
        if hint not in output_names:
            raise KeyError(
                f"classification_output {hint!r} is not an output of the model. "
                f"Actual model output names: {output_names}"
            )
        return hint
    taken = {f"{n}{assign_suffix}" for n in particle_names} | {
        f"{n}{detect_suffix}" for n in particle_names
    }
    rest = [
        n
        for n in output_names
        if n not in taken
        and not n.endswith((assign_suffix, detect_suffix))
        and "regress" not in n.lower()
    ]
    if len(rest) == 1:
        return rest[0]
    named = [n for n in rest if "classif" in n.lower()]
    if len(named) == 1:
        return named[0]
    raise KeyError(
        f"Could not tell which ONNX output holds the classification scores (candidates after removing "
        f"assignment/detection/regression outputs: {rest}). Actual model output names: {output_names}. "
        f"If the model has no classification output, the stock export may not include one; otherwise set "
        f"classification_output to the exact name."
    )


def _to_class_probabilities(raw, mode="auto"):
    """
    (n_events, n_classes) class probabilities from the raw classification output.
    mode: 'probabilities' (already softmax'ed; used as is), 'logits' (apply softmax), or 'auto':
    treat the array as probabilities if every entry is in [0, 1] and every row sums to 1 within
    1e-3, otherwise apply softmax. The decision is made once on the whole array, never per row.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(
            f"classification output must be 2D (n_events, n_classes), got shape {raw.shape}"
        )
    if mode not in ("auto", "logits", "probabilities"):
        raise ValueError(
            f"classification_scores must be 'auto', 'logits' or 'probabilities', got {mode!r}"
        )
    if mode == "auto":
        looks_prob = bool(
            raw.size == 0
            or (
                raw.min() >= 0.0
                and raw.max() <= 1.0
                and np.allclose(raw.sum(axis=1), 1.0, atol=1e-3)
            )
        )
        mode = "probabilities" if looks_prob else "logits"
    if mode == "logits":
        z = raw - raw.max(axis=1, keepdims=True)
        e = np.exp(z)
        raw = e / e.sum(axis=1, keepdims=True)
    return raw.astype("float32")


def _zero_index_all_axes(tensor, jet_index):
    """Zero out every slice of `tensor` where ANY axis equals jet_index --
    generalizes the original H1/H2 zeroing (which hand-wrote each axis
    explicitly) to work on a tensor of ANY rank, including ISR's rank-1
    (single-jet) case, without a separate branch per particle."""
    for axis in range(tensor.ndim):
        idx = [slice(None)] * tensor.ndim
        idx[axis] = jet_index
        tensor[tuple(idx)] = 0


def _extract_pairs_exclusive(assign_tensors):
    """
    Exclusive jet assignment, generalized from the original 2-particle
    (H1-vs-H2) version to an arbitrary number of particles: at each
    round, whichever REMAINING particle currently has the highest max
    confidence gets assigned next (via its own argmax); its claimed jet
    indices are then zeroed out of every OTHER remaining tensor
    (regardless of that tensor's rank) before the next round. Confirmed
    to exactly reproduce the original 2-particle algorithm's output when
    given only 2 tensors, and confirmed exclusive (no jet claimed by
    more than one particle) and exact-recovery-correct on synthetic
    3-particle test data with realistic, self-consistent target peaks.

    Parameters
    ----------
    assign_tensors : list of np.ndarray
        One array per particle, each shape (n_events,) + (n_jets,)*rank,
        where rank is that particle's number of daughter slots (H1=2,
        H2=4 for full_hww, ISR=1).

    Returns
    -------
    list of np.ndarray, one per particle, each shape (n_events, rank),
    in the SAME order as assign_tensors.
    """
    n_events = assign_tensors[0].shape[0]
    n_particles = len(assign_tensors)
    preds = [
        np.full((n_events, assign_tensors[k].ndim - 1), -1, dtype=int)
        for k in range(n_particles)
    ]

    for i in range(n_events):
        tensors = [a[i].copy() for a in assign_tensors]
        remaining = list(range(n_particles))

        while remaining:
            confidences = [np.max(tensors[k]) for k in remaining]
            winner = remaining.pop(int(np.argmax(confidences)))

            pred_idx = np.unravel_index(
                np.argmax(tensors[winner]), tensors[winner].shape
            )
            preds[winner][i] = pred_idx

            for jet_index in pred_idx:
                for k in remaining:
                    _zero_index_all_axes(tensors[k], jet_index)

    return preds


def _gather_assigned_jets(pt, eta, phi, mass, preds):
    """
    Gather each particle's jet collection from the padded source arrays,
    using the network's predicted indices -- generalized from the
    original H1(2)/H2(4) version to accept any number of particles with
    any daughter count, including ISR's single-jet (rank-1) case.

    Uses ak.from_regular to convert the numpy index arrays to the jagged
    ("var") form -- verified this is required: a regular (fixed-width)
    awkward index array silently applies the WRONG (flattened) indexing
    semantics against a jagged content array rather than erroring, so the
    from_regular conversion is not optional cosmetic cleanup, it is what
    makes the per-event gather correct at all.

    If a predicted assignment includes the null slot (index n_real_jets,
    zero-padded), that slot's four-vector is exactly (0,0,0,0) and
    contributes nothing to a sum -- confirmed this reproduces the exact
    same masses as the original per-event-loop implementation to
    float32 precision, so this is a drop-in numerical replacement, not a
    behavior change.

    Parameters
    ----------
    preds : list of np.ndarray
        One array per particle, each shape (n_events, rank), in the SAME
        order the caller wants results back in.

    Returns
    -------
    list of ak.Array, one PtEtaPhiMLorentzVector jet collection per particle,
    in the SAME order as preds.
    """
    p4 = ak.zip(
        {
            "pt": ak.Array(pt),
            "eta": ak.Array(eta),
            "phi": ak.Array(phi),
            "mass": ak.Array(mass),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=coffea_vector.behavior,
    )
    return [p4[ak.from_regular(ak.Array(pred), axis=1)] for pred in preds]


@define
class SPANetDiHiggsInference(AnalyzerModule):
    """
    Inference module for the SPANet HH->bbWW(+ISR) jet-assignment network.
    Takes jet-like columns, applies the btag-then-secondary jet ordering
    SPANet was trained on, prepares padded Source inputs, runs the ONNX
    model, and adds each particle's reconstructed jet collection (and,
    for H1/H2, summed mass) to the columns.

    Generalized to N particles via particle_names (default ["H1", "H2",
    "ISR"], matching the current 3-particle Hbb/HWW/ISR training) --
    outputs for EVERY name in particle_names get written, e.g. with the
    default list: H1_jets, H2_jets, ISR_jets, plus m_Hbb_SPANet (summed
    from H1_jets) and m_HWW_SPANet (summed from H2_jets). ISR has no
    analogous summed-mass output -- it's a single jet, not a pair/quad,
    so its own mass is just ISR_jets.mass directly, nothing to sum.

    Detection scores: for EVERY name in particle_names the network's detection
    probability (the ONNX "<name>_detection_probability" output, i.e. sigmoid of
    the detection logit -- the same quantity batch_trace.py calls
    detection_sigmoid) is also written, as "<name>_detection".
    These are what a detection-score cut or detection-binned region in OSCA
    should use; the assignment probabilities alone say nothing about whether
    the event looks like signal.

    Classification scores (optional, use_classification=True): for a model trained with a
    classification head (process_label, 0 = QCD, 1 = signal), the per-event class probabilities are
    written as "<prefix>.classification_<name>" for every name in classification_names (default
    ["qcd", "signal", "ttbar"], in class-label order), e.g. SPANet.classification_signal = P(signal).
    The output tensor is found by name (classification_output, or inferred -- see
    _find_classification_output); classification_scores says whether it holds logits or
    probabilities ("auto" checks). With use_classification=False (default) nothing about the
    classification head is read, so models exported without one keep working.

    Separate module from ABCDiHiggsInference (the ABCD background
    discriminant) -- different model, different purpose. Does not modify
    or depend on that class.

    Parameters
    ----------
    jet_col : Column
        Column containing the jet collection (already selected upstream:
        eta/pt cuts, trigger, >=5 jets).
    pt_field, eta_field, phi_field, mass_field, btag_field, qvg_field : str
        Field names for the jet kinematic/tagging variables. btag_field
        follows the same era-dependent-tagger resolution convention as
        ABCDiHiggsInference (pass "btag" to trigger it).
    model_path : str
        Path to the SPANet .onnx model.
    output_prefix : Column
        Column prefix under which m_Hbb_SPANet / m_HWW_SPANet / H*_jets
        are written.
    particle_names : list[str], optional
        Names of the particles the model was trained to predict, in
        whatever order the model's own ONNX outputs use internally
        (order here only affects the greedy exclusive-assignment
        priority resolution, not correctness -- see
        _extract_pairs_exclusive). Default ["H1", "H2", "ISR"].
    daughter_counts : dict[str, int], optional
        Number of daughter slots per particle name, needed to build
        correctly-shaped empty placeholders for the n_events==0 case.
        Default {"H1": 2, "H2": 4, "ISR": 1}.
    n_real_jets, n_null_jets : int, optional
        Must match the values the model was trained with.
    reco_mode : str, optional
        Kept for compatibility with older 2-particle configs; unused by
        the current N-particle assignment logic itself (rank is read
        directly from each ONNX output tensor's own shape instead).
    batch_size : int, optional
        Batched ONNX inference chunk size, default 256.
    use_classification : bool, optional
        Read the model's classification output and write the class-probability columns.
    classification_output : str, optional
        Exact ONNX output name of the classification scores; inferred if omitted.
    classification_names : list[str], optional
        One name per class, in class-label order (process_label 0, 1, 2, ...). Default
        ["qcd", "signal", "ttbar"] for the 3-class head; set ["qcd", "signal"] for a 2-class model.
        Writes <prefix>.classification_<name> per entry, so ttbar events get a classification_ttbar score too.
    classification_scores : str, optional
        "auto" (default), "logits" or "probabilities" -- what the ONNX output contains.
    """

    jet_col: Column
    pt_field: str
    eta_field: str
    phi_field: str
    mass_field: str
    btag_field: str
    qvg_field: str
    model_path: str
    output_prefix: Column
    n_real_jets: int = 6
    n_null_jets: int = 1
    reco_mode: str = "full_hww"
    batch_size: int = 256
    secondary_order_field: str = "qvg"
    particle_names: list = None
    daughter_counts: dict = None
    use_classification: bool = False
    classification_output: str = None
    classification_names: list = None
    classification_scores: str = "auto"

    btag_working_point: str = (
        "M"  # official WP ("L"/"M"/"T") used when btag_mode="label"
    )
    btag_mode: str = (
        "label"  # "score" (continuous discriminant) or "label" (boolean WP decision). Must match whichever mode the .onnx model was actually trained with.
    )
    qvg_mode: str = (
        "label"  # "score" or "label", applied to whichever qvg_field is configured (PNet QvG for any model trained after UParT QvG was removed from skimming.py)
    )
    qvg_label_threshold: float = (
        0.3  # UNOFFICIAL -- no correctionlib/official WP source exists for QvG. Must match skimming.py's value exactly for label-mode models.
    )
    ctag_working_point: str = "M"  # official WP used when ctag_mode="label"
    ctag_mode: str = (
        "label"  # "score" (2 independent columns: CvB, CvL), "label" (1 boolean column: CvB AND CvL both pass), or "none" (no ctag columns at all, and c-tagging metadata is never even resolved -- safe for checkpoints trained before c-tagging existed, e.g. the original region1-5 study). Changes Source WIDTH -- must match the specific .onnx model's actual trained shape, not just its physics content.
    )

    __wp_cache: dict = field(factory=dict)
    __ctag_wp_cache: dict = field(factory=dict)

    def __attrs_post_init__(self):
        if self.particle_names is None:
            self.particle_names = ["H1", "H2", "ISR"]
        if self.daughter_counts is None:
            self.daughter_counts = {"H1": 2, "H2": 4, "ISR": 1}
        if self.classification_names is None:
            # label values of the training h5 (process_label): 0 = QCD, 1 = signal, 2 = ttbar -- the current
            # 3-class head. A 2-class (QCD vs signal) model must set classification_names: [qcd, signal].
            self.classification_names = ["qcd", "signal", "ttbar"]

    def getWPs(self, metadata):
        """Mirrors HBQuarkMaker.getWPs / skimming.py's SPANetGenMatch.getWPs
        exactly -- same metadata path, same correctionlib call, same
        per-file caching, so inference reads WP thresholds from the SAME
        source training-time skimming.py used, not a second,
        independently-hardcoded number that could silently drift."""
        file_path = metadata["era"]["btag_scale_factors"]["file"]
        tagger = metadata["era"]["btag_scale_factors"]["tagger"]
        cname = metadata["era"]["btag_scale_factors"]["correction_name"]

        if file_path in self.__wp_cache:
            return tagger, self.__wp_cache[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = {p: cset[cname].evaluate(p) for p in ("L", "M", "T")}
        self.__wp_cache[file_path] = ret
        return tagger, ret

    def getCTagWPs(self, metadata):
        """c-tagging analog of getWPs, mirroring skimming.py's
        SPANetGenMatch.getCTagWPs exactly."""
        file_path = metadata["era"]["btag_scale_factors"]["c_file"]
        tagger_cvb = metadata["era"]["btag_scale_factors"]["c_tagger"]["CvB"]
        tagger_cvl = metadata["era"]["btag_scale_factors"]["c_tagger"]["CvL"]
        taggers = {"CvB": tagger_cvb, "CvL": tagger_cvl}
        cname = metadata["era"]["btag_scale_factors"]["correction_name"]

        if file_path in self.__ctag_wp_cache:
            return taggers, self.__ctag_wp_cache[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = {
            "CvB": {p: cset[cname].evaluate(p, "CvB") for p in ("L", "M", "T")},
            "CvL": {p: cset[cname].evaluate(p, "CvL") for p in ("L", "M", "T")},
        }
        self.__ctag_wp_cache[file_path] = ret
        return taggers, ret

    def preloadForMeta(self, metadata):
        self.getWPs(metadata)
        self.getCTagWPs(metadata)

    def prepare_inputs(self, columns):
        btag_col = Column(
            self.btag_field
            if self.btag_field != "btag"
            else columns.metadata["era"]["btag_scale_factors"]["tagger"]
        )
        pt = columns[self.jet_col + Column(self.pt_field)]
        eta = columns[self.jet_col + Column(self.eta_field)]
        phi = columns[self.jet_col + Column(self.phi_field)]
        mass = columns[self.jet_col + Column(self.mass_field)]
        p4 = ak.zip(
            {"pt": pt, "eta": eta, "phi": phi, "mass": mass},
            with_name="PtEtaPhiMLorentzVector",
            behavior=coffea_vector.behavior,
        )
        e = p4.energy

        raw_fields = {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "e": e,
            "mass": mass,
            "btag": columns[self.jet_col + btag_col],
            "qvg": columns[self.jet_col + Column(self.qvg_field)],
        }

        # ctag_mode="none": for checkpoints trained before c-tagging existed
        # (e.g. the original region1-5 study). This must not touch
        # c-tagging metadata AT ALL -- resolving it unconditionally would
        # fail outright on an era config that predates c-tagging support,
        # which is exactly the situation this mode exists to handle safely.
        if self.ctag_mode != "none":
            ctag_taggers, _ = self.getCTagWPs(columns.metadata)
            # Fetched into raw_fields alongside everything else so
            # _order_fields_btag_then_secondary reorders them with the SAME
            # sort permutation as pt/eta/phi/e/btag/qvg -- it iterates
            # field_arrays.items() generically, no changes needed there.
            raw_fields["ctag_cvb"] = columns[self.jet_col + Column(ctag_taggers["CvB"])]
            raw_fields["ctag_cvl"] = columns[self.jet_col + Column(ctag_taggers["CvL"])]

        if self.secondary_order_field not in ("qvg", "pt"):
            raise ValueError(
                f"secondary_order_field must be 'qvg' or 'pt', got "
                f"{self.secondary_order_field!r} -- these are the only two "
                f"fields genuinely fetched into raw_fields; anything else "
                f"would silently KeyError deep inside the sort."
            )
        sorted_fields = _order_fields_btag_then_secondary(
            raw_fields, self.secondary_order_field
        )

        # btag/qvg/ctag FEATURES fed to the model can each independently be
        # the continuous score or a boolean WP decision, per
        # btag_mode/qvg_mode/ctag_mode -- deliberately separate from the
        # sort above, which always used the continuous scores regardless
        # of mode. Only the copy AFTER sorting is touched here; mirrors
        # skimming.py's identical discipline exactly.
        if self.btag_mode == "label":
            _, wps = self.getWPs(columns.metadata)
            sorted_fields["btag"] = sorted_fields["btag"] > wps[self.btag_working_point]
        elif self.btag_mode != "score":
            raise ValueError(
                f"btag_mode must be 'score' or 'label', got {self.btag_mode!r}"
            )

        if self.qvg_mode == "label":
            sorted_fields["qvg"] = sorted_fields["qvg"] > self.qvg_label_threshold
        elif self.qvg_mode != "score":
            raise ValueError(
                f"qvg_mode must be 'score' or 'label', got {self.qvg_mode!r}"
            )

        extra_features = []
        if self.ctag_mode == "none":
            pass  # no ctag columns at all -- matches pre-ctagging checkpoints' 6-column Source exactly
        elif self.ctag_mode == "label":
            _, ctag_wps = self.getCTagWPs(columns.metadata)
            cvb_pass = (
                sorted_fields["ctag_cvb"] > ctag_wps["CvB"][self.ctag_working_point]
            )
            cvl_pass = (
                sorted_fields["ctag_cvl"] > ctag_wps["CvL"][self.ctag_working_point]
            )
            ctag_feature = cvb_pass & cvl_pass
            extra_features = [
                _pad_and_convert(ctag_feature, self.n_real_jets, self.n_null_jets)
            ]
        elif self.ctag_mode == "score":
            extra_features = [
                _pad_and_convert(
                    sorted_fields["ctag_cvb"], self.n_real_jets, self.n_null_jets
                ),
                _pad_and_convert(
                    sorted_fields["ctag_cvl"], self.n_real_jets, self.n_null_jets
                ),
            ]
        else:
            raise ValueError(
                f"ctag_mode must be 'score', 'label', or 'none', got {self.ctag_mode!r}"
            )

        mask = _make_mask(
            ak.num(sorted_fields["pt"]), self.n_real_jets, self.n_null_jets
        )
        source = {"MASK": mask, "_extra_features": extra_features}
        for key in ("pt", "eta", "phi", "e", "mass", "btag", "qvg"):
            source[key] = _pad_and_convert(
                sorted_fields[key], self.n_real_jets, self.n_null_jets
            )
        return source

    def _detection_cols(self):
        """Per-particle detection-score columns."""
        return {
            name: self.output_prefix + Column(f"{name}_detection")
            for name in self.particle_names
        }

    def _classification_cols(self):
        """Per-class probability columns (empty unless use_classification)."""
        if not self.use_classification:
            return {}
        return {
            name: self.output_prefix + Column(f"classification_{name}")
            for name in self.classification_names
        }

    def run(self, columns, params):
        n_events = len(columns[self.jet_col + Column(self.pt_field)])
        detect_cols = self._detection_cols()
        class_cols = self._classification_cols()

        _MASS_COL_NAMES = {
            "H1": "m_Hbb_SPANet",
            "H2": "m_HWW_SPANet",
            "W1": "m_W1_SPANet",
        }
        mass_cols = {
            name: self.output_prefix + Column(_MASS_COL_NAMES[name])
            for name in self.particle_names
            if name in _MASS_COL_NAMES
        }
        jets_cols = {
            name: self.output_prefix + Column(f"{name}_jets")
            for name in self.particle_names
        }

        if n_events == 0:
            empty = np.array([], dtype="float32")
            for name in mass_cols:
                columns[mass_cols[name]] = ak.Array(empty)
            for name in detect_cols:
                columns[detect_cols[name]] = ak.Array(empty)
            for name in class_cols:
                columns[class_cols[name]] = ak.Array(empty)
            empty_2d = np.zeros(
                (0, self.n_real_jets + self.n_null_jets), dtype="float32"
            )
            empty_p4 = ak.zip(
                {
                    "pt": ak.Array(empty_2d),
                    "eta": ak.Array(empty_2d),
                    "phi": ak.Array(empty_2d),
                    "mass": ak.Array(empty_2d),
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=coffea_vector.behavior,
            )
            for name in self.particle_names:
                width = self.daughter_counts[name]
                empty_idx = ak.from_regular(
                    ak.Array(np.zeros((0, width), dtype=int)), axis=1
                )
                columns[jets_cols[name]] = empty_p4[empty_idx]
            return columns, []

        source = self.prepare_inputs(columns)
        session = onnxruntime.InferenceSession(self.model_path)

        source_data = _build_source_data(
            source["pt"],
            source["eta"],
            source["phi"],
            source["e"],
            source["btag"],
            source["qvg"],
            extra_features=source["_extra_features"],
        )
        extras = {}
        if self.use_classification:
            cls_name = _find_classification_output(
                [o.name for o in session.get_outputs()],
                self.particle_names,
                hint=self.classification_output,
            )
            assign_detect, extras = _run_onnx_inference(
                session,
                source_data,
                source["MASK"],
                self.particle_names,
                batch_size=self.batch_size,
                extra_output_names=[cls_name],
            )
            probs = _to_class_probabilities(
                extras[cls_name], self.classification_scores
            )
            if probs.shape[1] != len(self.classification_names):
                raise ValueError(
                    f"the model's classification output {cls_name!r} has {probs.shape[1]} classes but "
                    f"classification_names has {len(self.classification_names)} entries "
                    f"({self.classification_names}); set classification_names to one name per class "
                    f"(3-class model: [qcd, signal, ttbar]; 2-class model: [qcd, signal])."
                )
            for k, name in enumerate(self.classification_names):
                columns[class_cols[name]] = ak.Array(probs[:, k])
        else:
            assign_detect = _run_onnx_inference(
                session,
                source_data,
                source["MASK"],
                self.particle_names,
                batch_size=self.batch_size,
            )
        assign_tensors = [assign_detect[name][0] for name in self.particle_names]
        preds = _extract_pairs_exclusive(assign_tensors)
        jets = _gather_assigned_jets(
            source["pt"], source["eta"], source["phi"], source["mass"], preds
        )
        jets_by_name = dict(zip(self.particle_names, jets))

        for name, jet_col in jets_by_name.items():
            columns[jets_cols[name]] = jet_col
            if name in mass_cols:
                columns[mass_cols[name]] = jet_col.sum(axis=1).mass

        # Detection probabilities (already sigmoid'ed by the ONNX export; one value per event per
        # particle). reshape handles an (n,) or (n, 1) output alike.
        detections = {}
        for name in self.particle_names:
            det = np.asarray(assign_detect[name][1], dtype="float32").reshape(
                n_events, -1
            )[:, 0]
            detections[name] = det
            columns[detect_cols[name]] = ak.Array(det)

        return columns, []

    def neededResources(self, metadata):
        return [self.model_path]

    def outputs(self, metadata):
        _MASS_COL_NAMES = {
            "H1": "m_Hbb_SPANet",
            "H2": "m_HWW_SPANet",
            "W1": "m_W1_SPANet",
        }
        outs = [
            self.output_prefix + Column(_MASS_COL_NAMES[name])
            for name in self.particle_names
            if name in _MASS_COL_NAMES
        ]
        outs += [
            self.output_prefix + Column(f"{name}_jets") for name in self.particle_names
        ]
        detect_cols = self._detection_cols()
        outs += list(detect_cols.values())
        outs += list(self._classification_cols().values())
        return outs

    def inputs(self, metadata):
        return [self.jet_col]


@define
class DiHiggsMassMultiplicitySplit(AnalyzerModule):
    """
    Splits an assigned H2 (HWW) jet collection's invariant mass into
    "threejet" and "fourjet" variants, based on how many REAL jets (as
    opposed to a null/missing slot) actually went into the assignment --
    mirroring `nontop2b_threejet`/`nontop2b_fourjet`
    variables (built the same way, minus the top-2-b jets), but computed
    from H2's own assignment for each method rather than a separately
    re-sorted leftover pool, since our jet collection is capped at 6 (the
    SPANet input-size constraint) -- H2 already *is* the complete leftover
    set, there is nothing further to select from a larger pool the way
    ABCD's unbounded goodJet collection needs.

    Works identically, with no branching, on either of the two H2 shapes
    this analysis produces:
      - SPANetDiHiggsInference's H2_jets: always exactly 4 gathered slots,
        one of which may be the network's null-slot pick (a genuine
        (0,0,0,0) four-vector, not a sentinel to special-case).
      - BaselineDiHiggsMasses's H2_jets: naturally ragged, 3 jets for a
        5-jet event or 4 for a 6-jet event, never padded.
    In both cases "count jets with pt != 0" gives the correct real-jet
    count: for the ragged case there's nothing to exclude (real jets are
    never exactly pt==0), and for the padded case it correctly excludes
    the null slot.

    Parameters
    ----------
    h2_jet_col : Column
        The H2 (HWW) jet collection column -- SPANetDiHiggsInference's or
        BaselineDiHiggsMasses's `<prefix>.H2_jets` output.
    output_prefix : Column
        Column prefix under which m_HWW_threejet / m_HWW_fourjet /
        n_real_H2_jets are written.
    """

    h2_jet_col: Column
    output_prefix: Column

    def run(self, columns, params):
        jets = columns[self.h2_jet_col]

        real_mask = jets.pt != 0
        n_real = ak.sum(real_mask, axis=1)
        mass = jets[real_mask].sum(axis=1).mass

        is_three = n_real == 3
        is_four = n_real == 4

        columns[self.output_prefix + Column("m_HWW_threejet")] = ak.where(
            is_three, mass, np.nan
        )
        columns[self.output_prefix + Column("m_HWW_fourjet")] = ak.where(
            is_four, mass, np.nan
        )
        columns[self.output_prefix + Column("n_real_H2_jets")] = n_real
        return columns, []

    def outputs(self, metadata):
        return [
            self.output_prefix + Column("m_HWW_threejet"),
            self.output_prefix + Column("m_HWW_fourjet"),
            self.output_prefix + Column("n_real_H2_jets"),
        ]

    def inputs(self, metadata):
        return [self.h2_jet_col]
