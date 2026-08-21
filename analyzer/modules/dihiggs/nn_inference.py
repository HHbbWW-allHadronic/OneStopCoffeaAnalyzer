
from coffea.ml_tools.torch_wrapper import torch_wrapper
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define, field
import awkward as ak
import numpy as np
import torch
import onnxruntime
import correctionlib
import vector


@define
class ABCDiHiggsInference(AnalyzerModule):
    """
    Inference module for the ABCDiHiggs neural network.
    This module takes in jet-like columns and global event features,
    prepares the inputs for the neural network, performs inference,
    and adds the output discriminators to the columns.

    Parameters
    ----------
    jet_col : Column
        Column containing the jet collection.
    jet_vars : list[str]
        List of jet variables to be used as inputs for the neural network.
    global_cols : list[Column]
        List of global event feature columns to be used as inputs for the neural network.
    model_path : str
        Path to the trained neural network model file.
    scaler_path : str
        Path to the scaler file used for input normalization.
    output_cols : list[Column]
        List of columns to store the output discriminators from the neural network.
    n_jets : int, optional
        Number of jets to prepare for the input, default is 6.
    """

    jet_col: Column
    jet_vars: list[str]
    global_cols: list[Column]
    model_path: str
    scaler_path: str
    output_cols: list[Column]
    n_jets: int = 6

    def prepare_inputs(self, columns):
        inputs = []
        for var in self.jet_vars:
            field = Column(
                var
                if (var != "btag")
                else columns.metadata["era"]["btag_scale_factors"]["tagger"]
            )
            in_var = columns[self.jet_col + field]
            padded = ak.pad_none(in_var, self.n_jets, clip=True)
            padded = ak.fill_none(padded, 0)
            inputs.append(padded)
        for global_col in self.global_cols:
            inputs.append(columns[global_col][:, np.newaxis])
        return ak.concatenate(inputs, axis=1)

    def run(self, columns, params):
        n_discs = len(self.output_cols)

        class ABCDiHiggsNetwork(torch_wrapper):
            def prepare_awkward(self, inputs):
                return [
                    ak.values_astype(inputs, "float32"),
                ], {}

            def postprocess_awkward(self, output, events):
                ret = {f"Disc{i}": output[:, i] for i in range(n_discs)}
                return ret

        model = ABCDiHiggsNetwork(self.model_path)
        X = self.prepare_inputs(columns)
        with open(self.scaler_path, "rb") as f:
            scaler = torch.load(f, map_location="cpu", weights_only=False)["scaler"]
        X = (X - scaler.mean_) / scaler.scale_

        if len(X) == 0:
            empty = np.array([], dtype="float32")
            for col in self.output_cols:
                columns[col + Column("sig")] = ak.Array(empty)
                columns[col + Column("qcd")] = ak.Array(empty)
                columns[col + Column("tt")] = ak.Array(empty)
            return columns, []

        outputs = model(X)
        for i, col in enumerate(self.output_cols):
            columns[col + Column("sig")] = outputs[f"Disc{i}"][:, 0]
            columns[col + Column("qcd")] = outputs[f"Disc{i}"][:, 1]
            columns[col + Column("tt")] = outputs[f"Disc{i}"][:, 2]
        return columns, []

    def neededResources(self, metadata):
        return [self.model_path, self.scaler_path]

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
    return np.array([
        [True] * min(n, n_real_jets) + [False] * max(0, n_real_jets - n) + [True] * n_null_jets
        for n in jet_counts
    ], dtype=bool)


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


def _run_onnx_inference(session, source_data, mask, particle_names, batch_size=256,
                         assign_suffix="_assignment_probability",
                         detect_suffix="_detection_probability"):
    """
    Batched ONNX call. Looks up each particle's assignment/detection
    output BY NAME (via session.get_outputs()), not by fixed position --
    more robust than a hardcoded positional order, and self-documenting
    if a name doesn't match what's expected.

    Returns: dict mapping particle_name -> (assign_array, detect_array).
    """
    output_names = [o.name for o in session.get_outputs()]
    expected_names = [f"{name}{assign_suffix}" for name in particle_names] + \
                      [f"{name}{detect_suffix}" for name in particle_names]
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
        out = session.run(output_names, {
            "Source_data": source_data[i:i + batch_size],
            "Source_mask": mask[i:i + batch_size],
        })
        for name, arr in zip(output_names, out):
            collected[name].append(arr)

    results = {}
    for name in particle_names:
        assign = np.concatenate(collected[f"{name}{assign_suffix}"], axis=0)
        detect = np.concatenate(collected[f"{name}{detect_suffix}"], axis=0)
        results[name] = (assign, detect)
    return results


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

            pred_idx = np.unravel_index(np.argmax(tensors[winner]), tensors[winner].shape)
            preds[winner][i] = pred_idx

            for jet_index in pred_idx:
                for k in remaining:
                    _zero_index_all_axes(tensors[k], jet_index)

    return preds


def _gather_assigned_jets(pt, eta, phi, e, preds):
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
    list of ak.Array, one Momentum4D jet collection per particle, in the
    SAME order as preds.
    """
    p4 = ak.zip(
        {"pt": ak.Array(pt), "eta": ak.Array(eta), "phi": ak.Array(phi), "energy": ak.Array(e)},
        with_name="Momentum4D",
    )
    p4 = ak.zip(
        {"pt": p4.pt, "eta": p4.eta, "phi": p4.phi, "energy": p4.energy, "mass": p4.mass},
        with_name="Momentum4D",
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

    btag_working_point: str = "M"  # official WP ("L"/"M"/"T") used when btag_mode="label"
    btag_mode: str = "label"  # "score" (continuous discriminant) or "label" (boolean WP decision). Must match whichever mode the .onnx model was actually trained with.
    qvg_mode: str = "label"  # "score" or "label", applied to whichever qvg_field is configured (PNet QvG for any model trained after UParT QvG was removed from skimming.py)
    qvg_label_threshold: float = 0.3  # UNOFFICIAL -- no correctionlib/official WP source exists for QvG. Must match skimming.py's value exactly for label-mode models.
    ctag_working_point: str = "M"  # official WP used when ctag_mode="label"
    ctag_mode: str = "label"  # "score" (2 independent columns: CvB, CvL), "label" (1 boolean column: CvB AND CvL both pass), or "none" (no ctag columns at all, and c-tagging metadata is never even resolved -- safe for checkpoints trained before c-tagging existed, e.g. the original region1-5 study). Changes Source WIDTH -- must match the specific .onnx model's actual trained shape, not just its physics content.

    __wp_cache: dict = field(factory=dict)
    __ctag_wp_cache: dict = field(factory=dict)

    def __attrs_post_init__(self):
        if self.particle_names is None:
            self.particle_names = ["H1", "H2", "ISR"]
        if self.daughter_counts is None:
            self.daughter_counts = {"H1": 2, "H2": 4, "ISR": 1}

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
        pt   = columns[self.jet_col + Column(self.pt_field)]
        eta  = columns[self.jet_col + Column(self.eta_field)]
        phi  = columns[self.jet_col + Column(self.phi_field)]
        mass = columns[self.jet_col + Column(self.mass_field)]
        p4 = ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass}, with_name="Momentum4D")
        e = p4.energy

        raw_fields = {
            "pt":   pt,
            "eta":  eta,
            "phi":  phi,
            "e":    e,
            "btag": columns[self.jet_col + btag_col],
            "qvg":  columns[self.jet_col + Column(self.qvg_field)],
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
        sorted_fields = _order_fields_btag_then_secondary(raw_fields, self.secondary_order_field)

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
            raise ValueError(f"btag_mode must be 'score' or 'label', got {self.btag_mode!r}")

        if self.qvg_mode == "label":
            sorted_fields["qvg"] = sorted_fields["qvg"] > self.qvg_label_threshold
        elif self.qvg_mode != "score":
            raise ValueError(f"qvg_mode must be 'score' or 'label', got {self.qvg_mode!r}")

        extra_features = []
        if self.ctag_mode == "none":
            pass  # no ctag columns at all -- matches pre-ctagging checkpoints' 6-column Source exactly
        elif self.ctag_mode == "label":
            _, ctag_wps = self.getCTagWPs(columns.metadata)
            cvb_pass = sorted_fields["ctag_cvb"] > ctag_wps["CvB"][self.ctag_working_point]
            cvl_pass = sorted_fields["ctag_cvl"] > ctag_wps["CvL"][self.ctag_working_point]
            ctag_feature = cvb_pass & cvl_pass
            extra_features = [_pad_and_convert(ctag_feature, self.n_real_jets, self.n_null_jets)]
        elif self.ctag_mode == "score":
            extra_features = [
                _pad_and_convert(sorted_fields["ctag_cvb"], self.n_real_jets, self.n_null_jets),
                _pad_and_convert(sorted_fields["ctag_cvl"], self.n_real_jets, self.n_null_jets),
            ]
        else:
            raise ValueError(f"ctag_mode must be 'score', 'label', or 'none', got {self.ctag_mode!r}")

        mask = _make_mask(ak.num(sorted_fields["pt"]), self.n_real_jets, self.n_null_jets)
        source = {"MASK": mask, "_extra_features": extra_features}
        for key in ("pt", "eta", "phi", "e", "btag", "qvg"):
            source[key] = _pad_and_convert(sorted_fields[key], self.n_real_jets, self.n_null_jets)
        return source
 
    def run(self, columns, params):
        source = self.prepare_inputs(columns)
        n_events = len(source["pt"])

        _MASS_COL_NAMES = {
            "H1": "m_Hbb_SPANet",
            "H2": "m_HWW_SPANet",
            "W1": "m_W1_SPANet",
        }
        mass_cols = {
            name: self.output_prefix + Column(_MASS_COL_NAMES[name])
            for name in self.particle_names if name in _MASS_COL_NAMES
        }
        jets_cols = {name: self.output_prefix + Column(f"{name}_jets") for name in self.particle_names}

        if n_events == 0:
            empty = np.array([], dtype="float32")
            for name in mass_cols:
                columns[mass_cols[name]] = ak.Array(empty)
            empty_2d = np.zeros((0, self.n_real_jets + self.n_null_jets), dtype="float32")
            empty_p4 = ak.zip(
                {
                    "pt": ak.Array(empty_2d), "eta": ak.Array(empty_2d),
                    "phi": ak.Array(empty_2d), "energy": ak.Array(empty_2d),
                    "mass": ak.Array(empty_2d),
                },
                with_name="Momentum4D",
            )
            for name in self.particle_names:
                width = self.daughter_counts[name]
                empty_idx = ak.from_regular(ak.Array(np.zeros((0, width), dtype=int)), axis=1)
                columns[jets_cols[name]] = empty_p4[empty_idx]
            return columns, []

        session = onnxruntime.InferenceSession(self.model_path)

        source_data = _build_source_data(
            source["pt"], source["eta"], source["phi"],
            source["e"], source["btag"], source["qvg"],
            extra_features=source["_extra_features"],
        )
        assign_detect = _run_onnx_inference(
            session, source_data, source["MASK"], self.particle_names, batch_size=self.batch_size
        )
        assign_tensors = [assign_detect[name][0] for name in self.particle_names]
        preds = _extract_pairs_exclusive(assign_tensors)
        jets = _gather_assigned_jets(
            source["pt"], source["eta"], source["phi"], source["e"], preds
        )
        jets_by_name = dict(zip(self.particle_names, jets))

        for name, jet_col in jets_by_name.items():
            columns[jets_cols[name]] = jet_col
            if name in mass_cols:
                columns[mass_cols[name]] = ak.sum(jet_col, axis=1).mass

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
            for name in self.particle_names if name in _MASS_COL_NAMES
        ]
        outs += [self.output_prefix + Column(f"{name}_jets") for name in self.particle_names]
        return outs

    def inputs(self, metadata):
        return [self.jet_col]

vector.register_awkward()

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
        mass = ak.sum(jets[real_mask], axis=1).mass

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
