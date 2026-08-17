from coffea.ml_tools.torch_wrapper import torch_wrapper
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define, field
import awkward as ak
import numpy as np
import torch
import onnxruntime
from functools import lru_cache
from coffea.nanoevents.methods import vector as coffea_vector


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


_BEHAVIOR = coffea_vector.behavior
_ASSIGN_SUFFIX = "_assignment_probability"

# Kinematic fields carried through the ordering step. Jets are stored in
# pt/eta/phi/mass everywhere; energy is never stored, only computed via
# `.energy`, because coffea rejects a record holding both mass and energy
# ("multiple temporal aliases present").
_KINEMATIC = ("pt", "eta", "phi", "mass")


@lru_cache(maxsize=None)
def _get_session(model_path: str, providers: tuple[str, ...]):
    """One ORT session per (model, providers) per process.

    Cached rather than stored on the module so the module stays trivially
    picklable for worker dispatch, and so the session is built once per
    worker instead of once per chunk.
    """
    return onnxruntime.InferenceSession(model_path, providers=list(providers))


def _default_providers() -> tuple[str, ...]:
    avail = onnxruntime.get_available_providers()
    cuda = ("CUDAExecutionProvider",) if "CUDAExecutionProvider" in avail else ()
    return cuda + ("CPUExecutionProvider",)


def _order_jets(fields: dict, secondary_key: str) -> dict:
    """Sort jets btag-descending for the first two slots, then `secondary_key`
    descending for the rest.

    `secondary_key` selects which already-fetched field determines the order.
    It does not change which fields are fed to the model: every field in
    `fields`, including the one used for ordering, is reordered identically and
    passed through as its own feature.
    """
    order = ak.argsort(fields["btag"], axis=1, ascending=False)
    lead = {k: v[order] for k, v in fields.items()}
    rest = ak.argsort(lead[secondary_key][:, 2:], axis=1, ascending=False) + 2
    return {k: ak.concatenate([v[:, :2], v[rest]], axis=1) for k, v in lead.items()}


def _pad_to_numpy(arr, n_real: int, n_slots: int) -> np.ndarray:
    """Clip to `n_real` jets, pad to `n_slots` with zeros, materialize."""
    return ak.to_numpy(
        ak.fill_none(ak.pad_none(arr[:, :n_real], n_slots, clip=True), 0)
    )


def _source_mask(counts, n_real: int, n_null: int) -> np.ndarray:
    """True for real jets present in the event and for every null slot."""
    counts = np.asarray(counts)[:, None]
    real = np.arange(n_real)[None, :] < counts
    null = np.ones((len(counts), n_null), dtype=bool)
    return np.concatenate([real, null], axis=1)


def _padded_jets(ordered: dict, n_real: int, n_slots: int):
    """One padded PtEtaPhiM collection, used for both the model input and the
    gathered output. Null slots are (pt, eta, phi, mass) = 0, whose computed
    energy is exactly 0 and which contribute nothing to a sum."""
    return ak.zip(
        {f: ak.Array(_pad_to_numpy(ordered[f], n_real, n_slots)) for f in _KINEMATIC},
        with_name="PtEtaPhiMLorentzVector",
        behavior=_BEHAVIOR,
    )


def _build_source_data(jets, btag, qvg) -> np.ndarray:
    """Build the model's Source_data tensor, shape (n_events, n_slots, 6).

    The model was trained on exactly this stacking:

        [log(pt + 1), eta, phi, log(energy + 1), btag, qvg]

    Do not reorder, rename, or insert. Energy is computed from the jets rather
    than stored, which is numerically identical to deriving it before padding:
    it is a deterministic function of (pt, eta, mass), and a null slot's
    (0, 0, 0, 0) gives exactly 0.0, matching the zero the padding inserts.
    """
    return np.stack(
        [
            np.log(ak.to_numpy(jets.pt) + 1),
            ak.to_numpy(jets.eta),
            ak.to_numpy(jets.phi),
            np.log(ak.to_numpy(jets.energy) + 1),
            btag,
            qvg,
        ],
        axis=-1,
    ).astype(np.float32)


def _zero_index_all_axes(tensor: np.ndarray, jet_index: int) -> None:
    """Zero every slice of `tensor` where any axis equals `jet_index`."""
    for axis in range(tensor.ndim):
        idx = [slice(None)] * tensor.ndim
        idx[axis] = jet_index
        tensor[tuple(idx)] = 0


def _assign_greedy_exclusive(assign_tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Greedy exclusive jet assignment, one round per particle.

    Each round, whichever remaining particle has the highest max confidence is
    assigned via its own argmax; the jets it claims are then zeroed out of every
    other remaining tensor before the next round.

    Note: this prevents two *particles* claiming the same jet. It does not
    prevent one particle claiming the same jet in two of its own daughter slots
    (an argmax landing on the tensor diagonal) -- preserved from the original
    for bit-identical results. Zeroing each tensor's diagonal before the argmax
    would close that gap.

    Parameters
    ----------
    assign_tensors : list of np.ndarray
        One per particle, shape (n_events,) + (n_slots,) * rank.

    Returns
    -------
    list of np.ndarray, shape (n_events, rank), same order as the input.
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


def _infer_assignments(
    session, source_data, mask, particle_names, batch_size: int, n_slots: int
) -> list[np.ndarray]:
    """Run the model in batches, reducing each batch to jet indices immediately.

    The H2 assignment tensor is (batch, n_slots**4) floats and only its argmax
    is ever used, so it is discarded before the next batch is fetched. Peak
    memory is batch_size * n_slots**4 * 4 bytes, not n_events * that.

    Detection probabilities are not requested -- nothing downstream reads them.
    """
    names = [p + _ASSIGN_SUFFIX for p in particle_names]
    available = [o.name for o in session.get_outputs()]
    missing = [n for n in names if n not in available]
    if missing:
        raise KeyError(
            f"Model has no outputs {missing}. Available: {available}. "
            f"Check `particle_names` against the model actually being loaded."
        )

    chunks = None
    for start in range(0, len(source_data), batch_size):
        stop = start + batch_size
        feed = {"Source_data": source_data[start:stop], "Source_mask": mask[start:stop]}
        try:
            tensors = session.run(names, feed)
        except Exception as exc:
            if chunks is not None:
                raise
            # SPANet bakes its jet-slot count into the graph, so a mismatch
            # surfaces as an opaque reshape failure deep in the encoder.
            raise RuntimeError(
                f"ONNX inference failed on the first batch with n_real_jets + "
                f"n_null_jets = {n_slots}. Check that against the model "
                f"({session._model_path if hasattr(session, '_model_path') else 'loaded model'}). "
                f"Original error: {exc}"
            ) from exc
        if chunks is None:
            got = tensors[0].shape[1]
            if got != n_slots:
                raise ValueError(
                    f"Model expects {got} jet slots, configured for {n_slots} "
                    f"(n_real_jets + n_null_jets). Fix the configuration to match "
                    f"the model."
                )
            chunks = [[] for _ in particle_names]
        for chunk, pred in zip(chunks, _assign_greedy_exclusive(tensors)):
            chunk.append(pred)

    return [np.concatenate(c, axis=0) for c in chunks]


def _gather_assigned_jets(jets, preds) -> list:
    """Gather each particle's assignment out of the padded jet collection.

    `ak.from_regular` is required: the predicted-index arrays are regular
    (fixed-width) numpy, and slicing the jet collection with them as-is does not
    apply per-event indexing.

    A prediction may include a null slot, whose four-vector is exactly
    (0, 0, 0, 0) and contributes nothing to a sum.
    """
    return [jets[ak.from_regular(ak.Array(pred), axis=1)] for pred in preds]


@define
class SPANetDiHiggsInference(AnalyzerModule):
    """SPANet jet-assignment inference for HH->bbWW(+ISR).

    Applies the btag-then-secondary jet ordering the model was trained on,
    builds the padded Source inputs, runs the ONNX model in batches, and writes
    each particle's assigned jet collection plus the configured summed masses.

    Jet four-vectors use coffea's ``PtEtaPhiMLorentzVector`` behavior, so the
    summed mass is ``jets.sum(axis=1).mass``. Note that coffea's vector
    behaviors provide ``.sum()`` as a method and do not register an ``ak.sum``
    reducer for records; the scikit-hep ``vector`` package is the other way
    round. Do not mix the two in one collection.

    Parameters
    ----------
    jet_col : Column
        Jet collection, already selected upstream (eta/pt cuts, trigger, njet).
    pt_field, eta_field, phi_field, mass_field, btag_field, qvg_field : str
        Field names on `jet_col`. Pass ``"btag"`` for `btag_field` to resolve
        the era-dependent tagger from metadata.
    model_path : str
        Path to the SPANet .onnx file.
    output_prefix : Column
        Prefix for all written columns.
    n_real_jets, n_null_jets : int
        Must match the model. Their sum is validated against the model's
        assignment tensor width on the first batch.
    batch_size : int
        Events per ONNX call. Also bounds peak assignment-tensor memory.
    secondary_order_field : {"qvg", "pt"}
        Which field orders slots 2 and beyond.
    particle_names : list[str]
        Particles the model predicts. Each needs a
        ``<name>_assignment_probability`` output.
    mass_outputs : dict[str, str]
        Particle name -> output column name for that particle's summed mass.
        Particles absent from this mapping still get a jet collection written.
    providers : tuple[str, ...] or None
        ONNXRuntime execution providers; None auto-selects CUDA if available.
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
    batch_size: int = 256
    secondary_order_field: str = "qvg"
    particle_names: list = field(factory=lambda: ["H1", "H2", "ISR"])
    mass_outputs: dict = field(
        factory=lambda: {"H1": "m_Hbb_SPANet", "H2": "m_HWW_SPANet"}
    )
    providers: tuple | None = None

    def __attrs_post_init__(self):
        if self.secondary_order_field not in ("qvg", "pt"):
            raise ValueError(
                f"secondary_order_field must be 'qvg' or 'pt', got "
                f"{self.secondary_order_field!r}"
            )
        unknown = set(self.mass_outputs) - set(self.particle_names)
        if unknown:
            raise ValueError(
                f"mass_outputs references unknown particles {sorted(unknown)}; "
                f"particle_names is {self.particle_names}"
            )

    @property
    def n_slots(self) -> int:
        return self.n_real_jets + self.n_null_jets

    def _session(self):
        return _get_session(self.model_path, self.providers or _default_providers())

    def _jets_col(self, name: str) -> Column:
        return self.output_prefix + Column(f"{name}_jets")

    def _mass_col(self, name: str) -> Column:
        return self.output_prefix + Column(self.mass_outputs[name])

    def prepare_inputs(self, columns) -> dict:
        """Order and pad the jets, returning the padded PtEtaPhiM collection plus
        the two non-kinematic model inputs and the source mask."""
        btag_col = Column(
            self.btag_field
            if self.btag_field != "btag"
            else columns.metadata["era"]["btag_scale_factors"]["tagger"]
        )
        pt = columns[self.jet_col + Column(self.pt_field)]
        eta = columns[self.jet_col + Column(self.eta_field)]
        phi = columns[self.jet_col + Column(self.phi_field)]
        mass = columns[self.jet_col + Column(self.mass_field)]

        ordered = _order_jets(
            {
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "mass": mass,
                "btag": columns[self.jet_col + btag_col],
                "qvg": columns[self.jet_col + Column(self.qvg_field)],
            },
            self.secondary_order_field,
        )

        return {
            "jets": _padded_jets(ordered, self.n_real_jets, self.n_slots),
            "btag": _pad_to_numpy(ordered["btag"], self.n_real_jets, self.n_slots),
            "qvg": _pad_to_numpy(ordered["qvg"], self.n_real_jets, self.n_slots),
            "MASK": _source_mask(
                ak.num(ordered["pt"]), self.n_real_jets, self.n_null_jets
            ),
        }

    def _write_empty(self, columns):
        empty = np.array([], dtype="float32")
        for name in self.mass_outputs:
            columns[self._mass_col(name)] = ak.Array(empty)

        shapes = {o.name: o.shape for o in self._session().get_outputs()}
        zeros = np.zeros((0, self.n_slots), dtype="float32")
        jets = ak.zip(
            {f: ak.Array(zeros) for f in _KINEMATIC},
            with_name="PtEtaPhiMLorentzVector",
            behavior=_BEHAVIOR,
        )
        for name in self.particle_names:
            rank = len(shapes[name + _ASSIGN_SUFFIX]) - 1
            idx = ak.from_regular(ak.Array(np.zeros((0, rank), dtype=int)), axis=1)
            columns[self._jets_col(name)] = jets[idx]
        return columns

    def run(self, columns, params):
        # len() on a virtual array reads no buffers, so an empty chunk costs
        # nothing beyond the (cached) session.
        if len(columns[self.jet_col + Column(self.pt_field)]) == 0:
            return self._write_empty(columns), []

        source = self.prepare_inputs(columns)
        source_data = _build_source_data(source["jets"], source["btag"], source["qvg"])

        preds = _infer_assignments(
            self._session(),
            source_data,
            source["MASK"],
            self.particle_names,
            self.batch_size,
            self.n_slots,
        )
        jets = dict(
            zip(
                self.particle_names,
                _gather_assigned_jets(source["jets"], preds),
            )
        )

        for name, jet_col in jets.items():
            columns[self._jets_col(name)] = jet_col
        for name in self.mass_outputs:
            columns[self._mass_col(name)] = jets[name].sum(axis=1).mass
        return columns, []

    def neededResources(self, metadata):
        return [self.model_path]

    def outputs(self, metadata):
        return [self._mass_col(n) for n in self.mass_outputs] + [
            self._jets_col(n) for n in self.particle_names
        ]

    def inputs(self, metadata):
        return [self.jet_col]
