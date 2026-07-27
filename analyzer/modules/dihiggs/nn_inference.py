from coffea.ml_tools.torch_wrapper import torch_wrapper
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define
import awkward as ak
import numpy as np
import torch
import onnxruntime
import vector


@define
class ABCDiHiggsInference(AnalyzerModule):
    r"""
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
    feature (confirmed this is the case here, not a hypothetical) --
    conflating "what determines the order" with "what gets stacked as a
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


def _build_source_data(pt, eta, phi, e, btag, qvg):
    """Exact transform the model expects: pt/e log1p'd, 6 features total."""
    return np.stack(
        [np.log(pt + 1), eta, phi, np.log(e + 1), btag, qvg],
        axis=-1,
    ).astype(np.float32)


def _run_onnx_inference(session, source_data, mask, batch_size=256):
    """Batched ONNX call. Fixed output order: H1_assign, H2_assign, H1_detect, H2_detect."""
    n = len(source_data)
    all_outputs = [[] for _ in range(4)]
    for i in range(0, n, batch_size):
        out = session.run(None, {
            "Source_data": source_data[i:i + batch_size],
            "Source_mask": mask[i:i + batch_size],
        })
        for j in range(4):
            all_outputs[j].append(out[j])
    return [np.concatenate(a, axis=0) for a in all_outputs]


def _extract_pairs_exclusive(H1_assign, H2_assign, reco_mode="full_hww"):
    """
    Exclusive jet assignment: whichever of H1/H2 has the higher-confidence
    argmax gets first pick of jets; those indices are zeroed out of the
    other tensor before it's argmax'd, so no jet is double-assigned.
    """
    n_events = H1_assign.shape[0]
    H1_pred = np.full((n_events, 2), -1)
    H2_pred = np.full((n_events, 2 if reco_mode == "onshell_w" else 4), -1)

    for i in range(n_events):
        h1 = H1_assign[i].copy()
        h2 = H2_assign[i].copy()

        if np.max(h1) >= np.max(h2):
            H1_pred[i] = np.unravel_index(np.argmax(h1), h1.shape)
            for jet_index in H1_pred[i]:
                if reco_mode == "onshell_w":
                    h2[jet_index, :] = 0
                    h2[:, jet_index] = 0
                else:
                    h2[jet_index, :, :, :] = 0
                    h2[:, jet_index, :, :] = 0
                    h2[:, :, jet_index, :] = 0
                    h2[:, :, :, jet_index] = 0
            H2_pred[i] = np.unravel_index(np.argmax(h2), h2.shape)
        else:
            H2_pred[i] = np.unravel_index(np.argmax(h2), h2.shape)
            for jet_index in H2_pred[i]:
                h1[jet_index, :] = 0
                h1[:, jet_index] = 0
            H1_pred[i] = np.unravel_index(np.argmax(h1), h1.shape)

    return H1_pred, H2_pred


def _gather_assigned_jets(pt, eta, phi, e, H1_pred, H2_pred):
    """
    Gather H1 (Hbb, 2 jets) and H2 (HWW, up to 4 jets) collections from the
    padded 7-slot source arrays, using the network's predicted indices.

    Uses ak.from_regular to convert the numpy index arrays to the jagged
    ("var") form -- verified this is required: a regular (fixed-width)
    awkward index array silently applies the WRONG (flattened) indexing
    semantics against a jagged content array rather than erroring, so the
    from_regular conversion is not optional cosmetic cleanup, it is what
    makes the per-event gather correct at all.

    If H2's predicted assignment includes the null slot (index n_real_jets,
    zero-padded), that slot's four-vector is exactly (0,0,0,0) and
    contributes nothing to a sum -- confirmed this reproduces the exact
    same masses as the original per-event-loop implementation (which also
    implicitly included any null-slot pick in its sum) to float32
    precision, so this is a drop-in numerical replacement, not a behavior
    change.
    """
    p4 = ak.zip(
        {"pt": ak.Array(pt), "eta": ak.Array(eta), "phi": ak.Array(phi), "energy": ak.Array(e)},
        with_name="Momentum4D",
    )
    h1_idx = ak.from_regular(ak.Array(H1_pred), axis=1)
    h2_idx = ak.from_regular(ak.Array(H2_pred), axis=1)
    return p4[h1_idx], p4[h2_idx]

@define
class SPANetDiHiggsInference(AnalyzerModule):
    """
    Inference module for the SPANet HH->bbWW jet-assignment network.
    Takes jet-like columns, applies the btag-then-PNet-QvG jet ordering
    SPANet was trained on, prepares padded Source inputs, runs the ONNX
    model, and adds the reconstructed H1 (Hbb) / H2 (HWW) masses to the
    columns.
 
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
        Column prefix under which m_Hbb_SPANet / m_HWW_SPANet are written.
    n_real_jets, n_null_jets : int, optional
        Must match the values the model was trained with (default 6, 1).
    reco_mode : str, optional
        "full_hww" or "onshell_w" -- must match the loaded model.
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
        if self.secondary_order_field not in ("qvg", "pt"):
            raise ValueError(
                f"secondary_order_field must be 'qvg' or 'pt', got "
                f"{self.secondary_order_field!r} -- these are the only two "
                f"fields genuinely fetched into raw_fields; anything else "
                f"would silently KeyError deep inside the sort."
            )
        sorted_fields = _order_fields_btag_then_secondary(raw_fields, self.secondary_order_field)
 
        mask = _make_mask(ak.num(sorted_fields["pt"]), self.n_real_jets, self.n_null_jets)
        source = {"MASK": mask}
        for key in ("pt", "eta", "phi", "e", "btag", "qvg"):
            source[key] = _pad_and_convert(sorted_fields[key], self.n_real_jets, self.n_null_jets)
        return source
 
    def run(self, columns, params):
        source = self.prepare_inputs(columns)
        n_events = len(source["pt"])
 
        m_Hbb_col = self.output_prefix + Column("m_Hbb_SPANet")
        m_HWW_col = self.output_prefix + Column("m_HWW_SPANet")
        h1_jets_col = self.output_prefix + Column("H1_jets")
        h2_jets_col = self.output_prefix + Column("H2_jets")
 
        if n_events == 0:
            empty = np.array([], dtype="float32")
            columns[m_Hbb_col] = ak.Array(empty)
            columns[m_HWW_col] = ak.Array(empty)
            empty_2d = np.zeros((0, self.n_real_jets + self.n_null_jets), dtype="float32")
            empty_h1_idx = ak.from_regular(ak.Array(np.zeros((0, 2), dtype=int)), axis=1)
            empty_h2_idx_width = 2 if self.reco_mode == "onshell_w" else 4
            empty_h2_idx = ak.from_regular(
                ak.Array(np.zeros((0, empty_h2_idx_width), dtype=int)), axis=1
            )
            empty_p4 = ak.zip(
                {
                    "pt": ak.Array(empty_2d), "eta": ak.Array(empty_2d),
                    "phi": ak.Array(empty_2d), "energy": ak.Array(empty_2d),
                },
                with_name="Momentum4D",
            )
            columns[h1_jets_col] = empty_p4[empty_h1_idx]
            columns[h2_jets_col] = empty_p4[empty_h2_idx]
            return columns, []
 
        session = onnxruntime.InferenceSession(self.model_path)
 
        source_data = _build_source_data(
            source["pt"], source["eta"], source["phi"],
            source["e"], source["btag"], source["qvg"],
        )
        H1_assign, H2_assign, H1_detect, H2_detect = _run_onnx_inference(
            session, source_data, source["MASK"], batch_size=self.batch_size
        )
        H1_pred, H2_pred = _extract_pairs_exclusive(H1_assign, H2_assign, reco_mode=self.reco_mode)
        H1_jets, H2_jets = _gather_assigned_jets(
            source["pt"], source["eta"], source["phi"], source["e"], H1_pred, H2_pred
        )
 
        columns[m_Hbb_col] = ak.sum(H1_jets, axis=1).mass
        columns[m_HWW_col] = ak.sum(H2_jets, axis=1).mass
        columns[h1_jets_col] = H1_jets
        columns[h2_jets_col] = H2_jets
        return columns, []

    def neededResources(self, metadata):
        return [self.model_path]
 
    def outputs(self, metadata):
        return [
            self.output_prefix + Column("m_Hbb_SPANet"),
            self.output_prefix + Column("m_HWW_SPANet"),
            self.output_prefix + Column("H1_jets"),
            self.output_prefix + Column("H2_jets"),
        ]
 
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
    this analysis produces (verified both give identical n_real/mass
    results on matched test data):
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
