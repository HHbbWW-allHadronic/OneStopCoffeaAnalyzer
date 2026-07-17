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


def _order_fields_btag_then_qvg(field_arrays):
    """
    Sort jets: btag descending for the first 2 slots (H1/bb candidates),
    then PNet quark-vs-gluon descending for the remaining slots (H2/WW
    candidates) -- the ordering SPANet was trained on, chosen to suppress
    ISR contamination in the H2 assignment. Every array in field_arrays is
    reordered consistently per event.
    """
    btag_sort_idx = ak.argsort(field_arrays["btag"], axis=1, ascending=False)
    partially_sorted = {k: v[btag_sort_idx] for k, v in field_arrays.items()}

    qvg_indices = ak.argsort(partially_sorted["qvg"][:, 2:], axis=1, ascending=False) + 2
    return {
        k: ak.concatenate([v[:, :2], v[qvg_indices]], axis=1)
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


def _masses_from_predictions(pt, eta, phi, e, H1_pred, H2_pred):
    """Build H1 (Hbb) / H2 (HWW) invariant masses from predicted jet indices."""
    fourvec = np.stack([pt, eta, phi, e], axis=-1)
    n_events = len(fourvec)
    m_H1 = np.full(n_events, np.nan)
    m_H2 = np.full(n_events, np.nan)

    for i in range(n_events):
        h1_idx = [j for j in H1_pred[i] if j >= 0]
        jets = [vector.obj(pt=fourvec[i, j, 0], eta=fourvec[i, j, 1],
                            phi=fourvec[i, j, 2], e=fourvec[i, j, 3]) for j in h1_idx]
        if len(jets) >= 2:
            total = jets[0]
            for j in jets[1:]:
                total = total + j
            m_H1[i] = total.m

        h2_idx = [j for j in H2_pred[i] if j >= 0]
        jets = [vector.obj(pt=fourvec[i, j, 0], eta=fourvec[i, j, 1],
                            phi=fourvec[i, j, 2], e=fourvec[i, j, 3]) for j in h2_idx]
        if len(jets) >= 2:
            total = jets[0]
            for j in jets[1:]:
                total = total + j
            m_H2[i] = total.m

    return m_H1, m_H2


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

    def prepare_inputs(self, columns):
        btag_col = Column(
            self.btag_field
            if self.btag_field != "btag"
            else columns.metadata["era"]["btag_scale_factors"]["tagger"]
        )
        pt  = columns[self.jet_col + Column(self.pt_field)]
        eta = columns[self.jet_col + Column(self.eta_field)]
        phi = columns[self.jet_col + Column(self.phi_field)]
        mass = columns[self.jet_col + Column(self.mass_field)]

        # SPANet's model was trained on energy, not mass. OSCA's jet objects
        # expose mass (see jet_vars: [pt, eta, phi, mass, btag] in the ABCD
        # config), not a raw energy column, so compute it here rather than
        # assume one exists: E = sqrt((pt*cosh(eta))^2 + m^2), verified
        # against a direct four-vector construction before use.
        e = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)

        raw_fields = {
            "pt":   pt,
            "eta":  eta,
            "phi":  phi,
            "e":    e,
            "btag": columns[self.jet_col + btag_col],
            "qvg":  columns[self.jet_col + Column(self.qvg_field)],
        }
        sorted_fields = _order_fields_btag_then_qvg(raw_fields)

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

        if n_events == 0:
            empty = np.array([], dtype="float32")
            columns[m_Hbb_col] = ak.Array(empty)
            columns[m_HWW_col] = ak.Array(empty)
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
        m_H1, m_H2 = _masses_from_predictions(
            source["pt"], source["eta"], source["phi"], source["e"], H1_pred, H2_pred
        )

        columns[m_Hbb_col] = ak.Array(m_H1)
        columns[m_HWW_col] = ak.Array(m_H2)
        return columns, []

    def neededResources(self, metadata):
        return [self.model_path]

    def outputs(self, metadata):
        return [
            self.output_prefix + Column("m_Hbb_SPANet"),
            self.output_prefix + Column("m_HWW_SPANet"),
        ]

    def inputs(self, metadata):
        return [self.jet_col]

vector.register_awkward()

@define
class BaselineDiHiggsMasses(AnalyzerModule):
    r"""
    Computes H1 (Hbb) and H2 (HWW) invariant masses using the SAME
    btag-then-PNet-QvG jet ordering SPANet's inputs are built from, but
    WITHOUT running any neural network: H1 = sum of the 2 highest-btag
    jets (ordering slots 0-1), H2 = sum of the remaining 4 jets in
    qvg-sorted order (slots 2-5). This is the naive/rule-based assignment
    SPANet is meant to improve on -- an important result to have alongside
    the SPANet masses for comparison, not a replacement for them.

    Mirrors SPANetDiHiggsInference's field resolution, era-dependent btag
    tagger resolution, and mass->energy computation exactly, so the two
    are directly comparable -- differing only in whether a trained
    assignment network refines the ordering-based guess versus taking the
    ordering itself as the assignment.

    H2/HWW mass gracefully degrades to fewer jets for events with exactly
    5 jets (allowed by the >=5 jet selection): p4 is a ragged array, not
    padded, so summing 3 available jets instead of 4 for those events --
    matching SPANet's own graceful handling of fewer real jets via its
    null-jet-slot mechanism, rather than artificially NaN-ing those events
    out. (An earlier version of this module did NaN them out via an
    n>=6 gate; that was removed after it was found to poison downstream
    plotting's axis-range calculation on full-statistics runs, and it also
    made this module *less* permissive than SPANet for no good reason --
    both now degrade gracefully in the same way.)

    Parameters
    ----------
    jet_col : Column
        The jet collection -- should match whatever jet_col
        SPANetDiHiggsInference uses in the same pipeline, for an
        apples-to-apples comparison.
    pt_field, eta_field, phi_field, mass_field, btag_field, qvg_field : str
        Same meaning as in SPANetDiHiggsInference.
    output_prefix : Column
        Column prefix under which m_Hbb_baseline / m_HWW_baseline are written.
    """

    jet_col: Column
    pt_field: str
    eta_field: str
    phi_field: str
    mass_field: str
    btag_field: str
    qvg_field: str
    output_prefix: Column

    def run(self, columns, params):
        btag_col = Column(
            self.btag_field
            if self.btag_field != "btag"
            else columns.metadata["era"]["btag_scale_factors"]["tagger"]
        )
        pt   = columns[self.jet_col + Column(self.pt_field)]
        eta  = columns[self.jet_col + Column(self.eta_field)]
        phi  = columns[self.jet_col + Column(self.phi_field)]
        mass = columns[self.jet_col + Column(self.mass_field)]
        e = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)  # verified against a direct 4-vector construction
        btag = columns[self.jet_col + btag_col]
        qvg  = columns[self.jet_col + Column(self.qvg_field)]

        # Same btag-then-qvg ordering used for SPANet's own inputs (see
        # spanet_dihiggs_inference.py's _order_fields_btag_then_qvg).
        btag_sort_idx = ak.argsort(btag, axis=1, ascending=False)
        pt2, eta2, phi2, e2, qvg2 = (
            arr[btag_sort_idx] for arr in (pt, eta, phi, e, qvg)
        )
        qvg_indices = ak.argsort(qvg2[:, 2:], axis=1, ascending=False) + 2

        def reorder(arr):
            return ak.concatenate([arr[:, :2], arr[qvg_indices]], axis=1)

        pt3, eta3, phi3, e3 = (reorder(arr) for arr in (pt2, eta2, phi2, e2))

        p4 = ak.zip(
            {"pt": pt3, "eta": eta3, "phi": phi3, "energy": e3},
            with_name="Momentum4D",
        )

        n = ak.num(p4, axis=1)

        h1_sum = ak.sum(p4[:, 0:2], axis=1)
        h2_sum = ak.sum(p4[:, 2:6], axis=1)

        # H1 always has >=2 jets available (guaranteed by the upstream >=5
        # jet selection), and H2 naturally degrades to fewer jets on its
        # own -- p4 is a ragged/jagged array, not padded, so p4[:, 2:6]
        # already clips to whatever's actually there (3 jets for a 5-jet
        # event, 4 for 6+), and ak.sum handles that correctly with no
        # special-casing needed. No NaN gate required: matches SPANet's
        # own graceful degradation for events with fewer real jets, and
        # avoids poisoning downstream plotting's axis-range calculation
        # with NaN values (confirmed this was happening: full-statistics
        # m_HWW_baseline plots came back with a corrupted axis range while
        # m_Hbb_baseline, which never had this gate's failure mode, did not).
        
	m_h1 = h1_sum.mass
        m_h2 = h2_sum.mass

        columns[self.output_prefix + Column("m_Hbb_baseline")] = m_h1
        columns[self.output_prefix + Column("m_HWW_baseline")] = m_h2
        return columns, []

    def neededResources(self, metadata):
        return []

    def outputs(self, metadata):
        return [
            self.output_prefix + Column("m_Hbb_baseline"),
            self.output_prefix + Column("m_HWW_baseline"),
        ]

    def inputs(self, metadata):
        return [self.jet_col]
