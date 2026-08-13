import uuid
import hashlib
from pathlib import Path

import awkward as ak
import correctionlib
from attrs import define, field

from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from analyzer.utils.file_tools import copyFile
from analyzer.utils.structure_tools import dictToDot, dotFormat
from typing import List
import numpy as np


@define
class SaveEventsH5(AnalyzerModule):
    """
    Analyzer module that serializes event-level data to HDF5 files for NN training.

    - Jet-level variables are written as (n_events, n_jets) datasets named
      "{jet_col}_{var}" (with '.' replaced by '_').
    - Event-level variables are written as (n_events,) datasets named after the
      Column (with '.' replaced by '_').

    Assumes it is run in a region where every event has at least `n_jets` jets in
    `jet_col`. Raises otherwise.

    Parameters
    ----------
    prefix : str
        Destination directory prefix where the output HDF5 files will be copied.
        (You said this will be the region name/path.)
    jet_col : Column
        Jet collection column (e.g. Column("goodJet")).
    jet_vars : list[str]
        Fields on the jet collection (e.g. ["pt","eta","phi","mass"]).
    n_jets : int
        Number of leading jets to write per event.
    event_vars : list[Column], optional
        Event-level columns to write (e.g. [Column("HT"), Column("b_dijet_12.mass")]).
    output_format : str, optional
        Filename template expanded with metadata fields plus file_id and uuid.
        Note: uuid is only used for the *local* temp filename unless you include
        "{uuid}" in this format.
    """

    prefix: str
    jet_col: Column
    jet_vars: list[str]
    n_jets: int
    event_vars: list[Column] = field(factory=list)
    output_format: str = (
        "{dataset_name}__{sample_name}__{file_id}"
        "__{chunk.event_start}_{chunk.event_stop}.h5"
    )

    def run(self, columns, params):
        file_id = (
            hashlib.md5((columns.metadata["chunk"]["file_path"]).encode())
            .hexdigest()
            .upper()
        )
        uid = str(uuid.uuid4())

        target_name = dotFormat(
            self.output_format,
            **dict(dictToDot(columns.metadata)),
            file_id=file_id,
            uuid=uid,
        )
        target = f"{self.prefix}/{target_name}"

        base = Path("localsaved")
        base.mkdir(exist_ok=True, parents=True)
        local_filename = base / f"{uid}.h5"

        try:
            import h5py

            jets = columns[self.jet_col]

            # Validate jet multiplicity before writing anything
            min_jets = ak.min(ak.num(jets))
            if min_jets is None:
                return columns, []

            min_jets = int(min_jets)
            if min_jets < self.n_jets:
                raise ValueError(
                    f"SaveH5: collection '{self.jet_col}' requires at least {self.n_jets} jets/event, "
                    f"but found an event with {min_jets} in chunk "
                    f"{columns.metadata['chunk']['file_path']} "
                    f"[{columns.metadata['chunk']['event_start']}, "
                    f"{columns.metadata['chunk']['event_stop']}]"
                )

            jet_prefix = str(self.jet_col).replace(".", "_")

            with h5py.File(local_filename, "w") as f:
                # Per-jet datasets: (n_events, n_jets)
                for v in self.jet_vars:
                    field_name = (
                        v
                        if (v != "btag")
                        else columns.metadata["era"]["btag_scale_factors"]["tagger"]
                    )
                    arr = jets[field_name][:, : self.n_jets]
                    data = ak.to_numpy(arr)
                    f.create_dataset(
                        f"{jet_prefix}_{v}",
                        data=data,
                        compression="gzip",
                    )

                # Event-level datasets: (n_events,)
                for c in self.event_vars:
                    arr = columns[c]
                    data = ak.to_numpy(arr)
                    name = str(c).replace(".", "_")
                    f.create_dataset(
                        name,
                        data=data,
                        compression="gzip",
                    )

            copyFile(local_filename, target)
        finally:
            local_filename.unlink(missing_ok=True)

        return columns, []

    def inputs(self, metadata):
        return [self.jet_col] + list(self.event_vars)

    def outputs(self, metadata):
        return []


# ===========================================================================================
# module versions of nano_to_h5_V2.py - main goal is to generate h5 files for training SPANet
# ===========================================================================================

W_MASS_PDG = 80.377  # GeV

# ============================================================
# Matching + target-construction helpers
# Ported verbatim (logic unchanged) from nano_to_h5_V2.py's
# greedy_match / pad_and_convert / make_mask / make_targets.
# Only the calling convention around them changes to fit AnalyzerModule.
# ============================================================

def greedy_match(objects_a, objects_b, signal_field, dr_threshold=0.4, store_pt=False):
    max_a = int(ak.max(ak.num(objects_a)))
    max_b = int(ak.max(ak.num(objects_b)))

    eta_a_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_a.eta, max_a, clip=True), 0.0))
    phi_a_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_a.phi, max_a, clip=True), 0.0))
    eta_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_b.eta, max_b, clip=True), 0.0))
    phi_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_b.phi, max_b, clip=True), 0.0))
    sig_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(getattr(objects_b, signal_field), max_b, clip=True), -1))
    counts_a = ak.to_numpy(ak.num(objects_a))
    counts_b = ak.to_numpy(ak.num(objects_b))

    if store_pt:
        pt_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_b.pt, max_b, clip=True), np.nan))

    labels_out = []
    pt_out = [] if store_pt else None

    for i in range(len(counts_a)):
        n_a = int(counts_a[i])
        n_b = int(counts_b[i])

        labels = [-1] * n_a
        pts = [np.nan] * n_a

        if n_a == 0 or n_b == 0:
            labels_out.append(labels)
            if store_pt:
                pt_out.append(pts)
            continue

        eta_a = eta_a_np[i, :n_a]
        phi_a = phi_a_np[i, :n_a]
        eta_b = eta_b_np[i, :n_b]
        phi_b = phi_b_np[i, :n_b]

        deta = eta_a[:, None] - eta_b[None, :]
        dphi = phi_a[:, None] - phi_b[None, :]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dr_matrix = np.sqrt(deta ** 2 + dphi ** 2)

        assigned_a = set()
        assigned_b = set()

        while True:
            dr_masked = dr_matrix.copy()
            if assigned_a:
                dr_masked[list(assigned_a), :] = np.inf
            if assigned_b:
                dr_masked[:, list(assigned_b)] = np.inf
            if dr_masked.min() > dr_threshold:
                break
            ia, ib = np.unravel_index(dr_masked.argmin(), dr_masked.shape)
            labels[ia] = int(sig_b_np[i, ib])
            if store_pt:
                pts[ia] = float(pt_b_np[i, ib])
            assigned_a.add(ia)
            assigned_b.add(ib)

        labels_out.append(labels)
        if store_pt:
            pt_out.append(pts)

    if store_pt:
        return labels_out, pt_out
    return labels_out


def pad_and_convert(arr, max_real_jets, n_null_jets):
    padded = ak.pad_none(arr[:, :max_real_jets], max_real_jets + n_null_jets, clip=True)
    padded = ak.fill_none(padded, 0)
    return ak.to_numpy(padded)


def make_mask(jet_counts, max_real_jets, n_null_jets):
    return np.array([
        [True] * min(n, max_real_jets) + [False] * max(0, max_real_jets - n) + [True] * n_null_jets
        for n in jet_counts
    ], dtype=bool)


def make_targets(signal_arr, max_real_jets, n_null_jets, reco_mode, include_isr_target=True):
    n_events = len(signal_arr)
    null_idx = max_real_jets if n_null_jets > 0 else -1

    H1 = np.full((n_events, 2), null_idx, dtype=int)
    H2 = np.full((n_events, 4), null_idx, dtype=int)
    W1 = np.full((n_events, 2), null_idx, dtype=int)
    if include_isr_target:
        ISR = np.full((n_events, 1), null_idx, dtype=int)

    for i in range(n_events):
        h1_jets = np.where(signal_arr[i] == 1)[0]
        h2_jets = np.where(signal_arr[i] == 2)[0]
        h3_jets = np.where(signal_arr[i] == 3)[0]

        H1[i, :min(len(h1_jets), 2)] = h1_jets[:2]
        if include_isr_target:
            h_isr_jets = np.where(signal_arr[i] == 4)[0]
            ISR[i, :min(len(h_isr_jets), 1)] = h_isr_jets[:1]

        if reco_mode == "full_hww":
            W1[i, :min(len(h2_jets), 2)] = h2_jets[:2]
            all_q = np.concatenate([h2_jets[:2], h3_jets[:2]])
            H2[i, :min(len(all_q), 4)] = all_q[:4]
        elif reco_mode == "onshell_w":
            W1[i, :min(len(h2_jets), 2)] = h2_jets[:2]

        seen = set()
        for slot in range(2):
            if H1[i, slot] in seen:
                H1[i, slot] = -1
            else:
                seen.add(int(H1[i, slot]))

        if reco_mode == "full_hww":
            for slot in range(4):
                if H2[i, slot] in seen:
                    H2[i, slot] = -1
                else:
                    seen.add(int(H2[i, slot]))
        elif reco_mode == "onshell_w":
            for slot in range(2):
                if W1[i, slot] in seen:
                    W1[i, slot] = -1
                else:
                    seen.add(int(W1[i, slot]))

        # ISR dedup against H1/H2/W1 only matters (and is only computed at
        # all) when an ISR target is actually being built -- with
        # include_isr_target=False, an ISR-matched jet (signal==4) is never
        # dedup'd out of H1/H2/W1 either, since it was never a candidate for
        # those in the first place (h1_jets/h2_jets/h3_jets only ever draw
        # from signal==1/2/3). Nothing here changes H1/H2/W1's own values.
        if include_isr_target:
            if ISR[i, 0] in seen:
                ISR[i, 0] = -1
            else:
                seen.add(int(ISR[i, 0]))

    if reco_mode == "full_hww":
        out = {
            "H1": {"b1": H1[:, 0], "b2": H1[:, 1]},
            "H2": {"q1": H2[:, 0], "q2": H2[:, 1], "q3": H2[:, 2], "q4": H2[:, 3]},
        }
    elif reco_mode == "onshell_w":
        out = {
            "H1": {"b1": H1[:, 0], "b2": H1[:, 1]},
            "W1": {"q1": W1[:, 0], "q2": W1[:, 1]},
        }
    if include_isr_target:
        out["ISR"] = {"g1": ISR[:, 0]}
    return out


# ============================================================
# Module 1: gen-level matching + target/source construction
# ============================================================

@define
class SPANetGenMatch(AnalyzerModule):
    """
    Gen-matches jets to build the {signal-type, target-index} truth used to
    train SPANet, and packages both the network's Source inputs and its
    TARGETS into new Columns.

    Must run downstream of whatever selection chain (jetID, lepton veto, 2b
    requirement, tiered-pt, njets) you want the training population to match
    at inference time -- this module does no event selection of its own. It
    never declares outputs()=="EVENTS", so it can't drop rows; the
    `good_event` boolean it writes is meant to be consumed by a downstream
    SelectOnColumns step in the yaml (matching the existing preselection/
    selection pattern), not applied internally here.
    """

    jet_col: Column
    genpart_col: Column = field(factory=lambda: Column("GenPart"))  # ASSUMPTION -- confirm
    genjet_col: Column = field(factory=lambda: Column("GenJet"))    # ASSUMPTION -- confirm
    output_prefix: str = "SPANet"
    reco_mode: str = "full_hww"  # "full_hww" or "onshell_w"
    n_real_jets: int = 9
    n_null_jets: int = 1
    dr_threshold: float = 0.4
    genjet_eta_cut: float = 2.4
    genjet_pt_cut: float = 15.0
    secondary_order_field: str = "pt"  # "pt" or "qvg" -- ordering applied to jets after the leading 2 btag-sorted slots
    include_isr_target: bool = True  # False = Option A: still gen-match/label ISR jets (signal==4), but don't build a Targets.ISR group for SPANet to predict against

    btag_working_point: str = "M"  # official WP ("L"/"M"/"T") used when btag_mode="label", resolved via getWPs() -- same official correctionlib source HBQuarkMaker's event-selection cut uses
    btag_mode: str = "label"  # "score" (continuous discriminant) or "label" (boolean WP decision). Does not affect jet ORDERING, which always uses the continuous score regardless of this setting (see run(), sorting happens before this is applied).

    qvg_mode: str = "label"  # "score" or "label", applied to PNet QvG (the only QvG discriminant kept -- UParT QvG has been removed entirely: not useful, and ROC-curve testing showed its candidate WPs were unreasonable)
    qvg_label_threshold: float = 0.3  # UNOFFICIAL -- no correctionlib/official WP source exists for QvG as of writing. CONFIRMED for PNet QvG specifically.

    ctag_working_point: str = "M"  # official WP ("L"/"M"/"T") used when ctag_mode="label", resolved via getCTagWPs()
    ctag_mode: str = "label"  # "score" or "label". SCORE: two independent continuous features (CvB, CvL) are kept, since they're genuinely distinct discriminants. LABEL: collapses to ONE boolean feature, true only if the jet clears BOTH the CvB and CvL working points simultaneously. Because the number of Source columns differs between these two modes (2 vs 1), a single event_info.yaml can't describe both -- use two separate configs, one per mode, matching this project's existing pattern of two-config comparisons.

    __wp_cache: dict = field(factory=dict)
    __ctag_wp_cache: dict = field(factory=dict)

    def getWPs(self, metadata):
        """
        Mirrors HBQuarkMaker.getWPs exactly (same file/tagger/correction_name
        metadata path, same correctionlib call, same per-file caching) --
        this module reads its own working-point thresholds from the SAME
        source HBQuarkMaker's event-selection cut already uses, rather than
        a second, independently-hardcoded number that could silently drift
        out of sync with it.
        """
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
        """
        c-tagging analog of getWPs. Follows the same metadata/correctionlib
        pattern as HCQuarkMaker, resolving TWO independent discriminants
        (CvB, CvL) rather than one -- each gets its own tagger field name
        and its own set of L/M/T thresholds, evaluated separately (CvB
        evaluated as "CvB", CvL evaluated as "CvL" -- these are genuinely
        different correctionlib queries, not the same query reused twice).
        """
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

    def inputs(self, metadata):
        return [self.jet_col, self.genpart_col, self.genjet_col]

    def outputs(self, metadata):
        p = self.output_prefix
        outs = [Column(f"{p}.Source"), Column(f"{p}.good_event"), Column(("Selection", "good_event"))]
        if self.reco_mode == "full_hww":
            outs += [Column(f"{p}.Targets.H1"), Column(f"{p}.Targets.H2")]
        else:
            outs += [Column(f"{p}.Targets.H1"), Column(f"{p}.Targets.W1")]
        if self.include_isr_target:
            outs.append(Column(f"{p}.Targets.ISR"))
        return outs

    def run(self, columns, params):
        jets = columns[self.jet_col]
        genpart = columns[self.genpart_col]
        genjet = columns[self.genjet_col]

        all_gp_idx = ak.local_index(genpart, axis=1)

        # -- Truth object construction (ported from nano_to_h5_V2.py) --
        H_idx = all_gp_idx[(genpart.pdgId == 25) & genpart.hasFlags(["isLastCopy"])]
        b_mask = (
            (abs(genpart.pdgId) == 5)
            & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
            & ak.any(genpart.genPartIdxMother[:, :, None] == H_idx[:, None, :], axis=2)
        )
        b_quarks = ak.with_field(genpart[b_mask], 1, "signal")

        incoming_mask = (genpart.status == 21) & (genpart.genPartIdxMother == -1)
        incoming_idx = all_gp_idx[incoming_mask]
        isr_mask = (
            (genpart.pdgId == 21)
            & ak.any(genpart.genPartIdxMother[:, :, None] == incoming_idx[:, None, :], axis=2)
            & (genpart.status != 21)
        )
        isr_gluon = ak.with_field(genpart[isr_mask], 4, "signal")

        if self.reco_mode == "full_hww":
            W_mask = (abs(genpart.pdgId) == 24) & genpart.hasFlags(["isLastCopy"])
            W_all_idx = all_gp_idx[W_mask]
            W_bosons = genpart[W_mask]
            W_mass_1 = ak.flatten(W_bosons[:, :1].mass)
            W_mass_2 = ak.flatten(W_bosons[:, 1:2].mass)
            onshell_is_first = W_mass_1 >= W_mass_2
            onshell_W_idx = ak.where(onshell_is_first[:, None], W_all_idx[:, :1], W_all_idx[:, 1:2])
            offshell_W_idx = ak.where(onshell_is_first[:, None], W_all_idx[:, 1:2], W_all_idx[:, :1])

            q_onshell_mask = (
                (abs(genpart.pdgId) <= 4)
                & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
                & ak.any(genpart.genPartIdxMother[:, :, None] == onshell_W_idx[:, None, :], axis=2)
            )
            q_onshell = ak.with_field(genpart[q_onshell_mask], 2, "signal")

            q_offshell_mask = (
                (abs(genpart.pdgId) <= 4)
                & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
                & ak.any(genpart.genPartIdxMother[:, :, None] == offshell_W_idx[:, None, :], axis=2)
            )
            q_offshell = ak.with_field(genpart[q_offshell_mask], 3, "signal")

            chs = ak.concatenate([b_quarks, q_onshell, q_offshell, isr_gluon], axis=1)

            def good_event_mask(gj_with_signal):
                return (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 1, False), axis=1) >= 2
                ) & (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 2, False), axis=1)
                    + ak.sum(ak.fill_none(gj_with_signal.signal == 3, False), axis=1)
                    >= 3
                )
        else:  # onshell_w
            W_mask = (abs(genpart.pdgId) == 24) & genpart.hasFlags(["isLastCopy"])
            W_all_idx = all_gp_idx[W_mask]
            W_bosons = genpart[W_mask]
            W_mass_1 = ak.flatten(W_bosons[:, :1].mass)
            W_mass_2 = ak.flatten(W_bosons[:, 1:2].mass)
            onshell_is_first = W_mass_1 >= W_mass_2
            onshell_W_idx = ak.where(onshell_is_first[:, None], W_all_idx[:, :1], W_all_idx[:, 1:2])

            q_onshell_mask = (
                (abs(genpart.pdgId) <= 4)
                & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
                & ak.any(genpart.genPartIdxMother[:, :, None] == onshell_W_idx[:, None, :], axis=2)
            )
            q_from_W = ak.with_field(genpart[q_onshell_mask], 2, "signal")
            chs = ak.concatenate([b_quarks, q_from_W, isr_gluon], axis=1)

            def good_event_mask(gj_with_signal):
                return (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 1, False), axis=1) >= 2
                ) & (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 2, False), axis=1) >= 2
                )

        # -- Gen jet selection (same eta/pt floor as nano_to_h5_V2.py) --
        genjet_cut = (abs(genjet.eta) < self.genjet_eta_cut) & (genjet.pt > self.genjet_pt_cut)
        gen_gj = genjet[genjet_cut]

        # -- Stage 1: gen parton -> gen jet --
        signal_genjet_list = greedy_match(gen_gj, chs, "signal", dr_threshold=self.dr_threshold)
        gen_gj = ak.with_field(gen_gj, ak.Array(signal_genjet_list), "signal")
        signal_gen_jets = gen_gj[ak.Array([[s != -1 for s in ev] for ev in signal_genjet_list])]

        # -- Stage 2: gen jet -> reco jet, with pt_ratio sanity check --
        signal_reco_list, matched_pt_list = greedy_match(
            jets, signal_gen_jets, "signal", dr_threshold=self.dr_threshold, store_pt=True
        )
        jets = ak.with_field(jets, ak.Array(signal_reco_list), "signal")
        jets = ak.with_field(jets, ak.Array(matched_pt_list), "matched_genjet_pt")
        pt_ratio = jets.pt / jets.matched_genjet_pt
        good_match_ptr = ak.fill_none(
            (pt_ratio > 0.5) & (pt_ratio < 2.0) & (jets.signal != -1), False
        )
        # Jets failing the pt_ratio check get relabeled to -1 (background)
        # rather than dropped, keeping the jet collection's width intact.
        jets = ak.with_field(jets, ak.where(good_match_ptr, jets.signal, -1), "signal")

        # -- btag+secondary ordering, same convention used at inference time --
        btag_sort_idx = ak.argsort(jets.btagUParTAK4B, axis=1, ascending=False)
        jets_partial = jets[btag_sort_idx]

        if self.secondary_order_field == "pt":
            secondary_field = jets_partial[:, 2:].pt
        elif self.secondary_order_field == "qvg":
            secondary_field = jets_partial[:, 2:].btagPNetQvG
        else:
            raise ValueError(
                f"secondary_order_field must be 'pt' or 'qvg', got {self.secondary_order_field!r}"
            )

        pt_idx = ak.argsort(secondary_field, axis=1, ascending=False) + 2
        jets_sorted = ak.concatenate([jets_partial[:, :2], jets_partial[pt_idx]], axis=1)

        # FIXED: good_event was previously evaluated on the full, untruncated
        # `jets` collection -- but signal_arr (and therefore make_targets)
        # only ever sees the top n_real_jets after this same sort, via
        # pad_and_convert's arr[:, :max_real_jets] truncation. A genuine 3rd
        # q-jet ranked 7th-or-lower by btag+secondary order would pass
        # good_event (which saw it) but never reach make_targets (which
        # didn't) -- leaving H2 with only 2 real slots filled, both
        # remaining slots colliding on the single null_idx, and the second
        # one forced to -1. Evaluating good_event on the SAME truncated
        # window signal_arr actually uses closes that gap: any event that
        # passes genuinely has >=3 real q-jets among the 6 the network will
        # actually be given, not just somewhere in the full event.
        good_event = good_event_mask(jets_sorted[:, : self.n_real_jets])

        pt_arr = pad_and_convert(jets_sorted.pt, self.n_real_jets, self.n_null_jets)
        eta_arr = pad_and_convert(jets_sorted.eta, self.n_real_jets, self.n_null_jets)
        phi_arr = pad_and_convert(jets_sorted.phi, self.n_real_jets, self.n_null_jets)
        e_arr = pad_and_convert(jets_sorted.energy, self.n_real_jets, self.n_null_jets)

        if self.btag_mode == "label":
            tagger, wps = self.getWPs(columns.metadata)
            btag_feature = jets_sorted[tagger] > wps[self.btag_working_point]
        elif self.btag_mode == "score":
            btag_feature = jets_sorted.btagUParTAK4B
        else:
            raise ValueError(f"btag_mode must be 'score' or 'label', got {self.btag_mode!r}")
        btag_arr = pad_and_convert(btag_feature, self.n_real_jets, self.n_null_jets)

        if self.qvg_mode == "label":
            pnet_qvg_feature = jets_sorted.btagPNetQvG > self.qvg_label_threshold
        elif self.qvg_mode == "score":
            pnet_qvg_feature = jets_sorted.btagPNetQvG
        else:
            raise ValueError(f"qvg_mode must be 'score' or 'label', got {self.qvg_mode!r}")
        pnet_qvg_arr = pad_and_convert(pnet_qvg_feature, self.n_real_jets, self.n_null_jets)

        # c-tagging: SCORE mode keeps CvB and CvL as two independent
        # continuous features (they're genuinely distinct discriminants,
        # not interchangeable). LABEL mode collapses them into ONE boolean
        # feature -- true only if the jet clears BOTH working points
        # simultaneously. This means Source has a DIFFERENT NUMBER of
        # columns depending on ctag_mode -- intentional, and why this needs
        # two separate event_info.yaml configs rather than one that tries
        # to describe both.
        ctag_source_fields = {}
        if self.ctag_mode == "label":
            taggers, wps = self.getCTagWPs(columns.metadata)
            cvb_pass = jets_sorted[taggers["CvB"]] > wps["CvB"][self.ctag_working_point]
            cvl_pass = jets_sorted[taggers["CvL"]] > wps["CvL"][self.ctag_working_point]
            ctag_feature = cvb_pass & cvl_pass
            ctag_source_fields["ctag"] = pad_and_convert(ctag_feature, self.n_real_jets, self.n_null_jets)
        elif self.ctag_mode == "score":
            taggers, _ = self.getCTagWPs(columns.metadata)
            ctag_source_fields["ctag_cvb"] = pad_and_convert(
                jets_sorted[taggers["CvB"]], self.n_real_jets, self.n_null_jets
            )
            ctag_source_fields["ctag_cvl"] = pad_and_convert(
                jets_sorted[taggers["CvL"]], self.n_real_jets, self.n_null_jets
            )
        else:
            raise ValueError(f"ctag_mode must be 'score' or 'label', got {self.ctag_mode!r}")

        signal_arr = pad_and_convert(jets_sorted.signal, self.n_real_jets, self.n_null_jets)

        jet_counts = ak.num(jets_sorted)
        mask = make_mask(jet_counts, self.n_real_jets, self.n_null_jets)

        source_fields = {
            "MASK": mask, "pt": pt_arr, "eta": eta_arr, "phi": phi_arr, "e": e_arr,
            "btag": btag_arr, "pnet_qvg": pnet_qvg_arr,
        }
        source_fields.update(ctag_source_fields)
        source = ak.zip(source_fields, depth_limit=1)

        targets = make_targets(
            signal_arr, self.n_real_jets, self.n_null_jets, self.reco_mode,
            include_isr_target=self.include_isr_target,
        )

        p = self.output_prefix
        columns[Column(f"{p}.Source")] = source
        columns[Column(f"{p}.good_event")] = good_event
        columns[Column(("Selection", "good_event"))] = good_event
        columns[Column(f"{p}.Targets.H1")] = ak.zip(targets["H1"])
        if self.reco_mode == "full_hww":
            columns[Column(f"{p}.Targets.H2")] = ak.zip(targets["H2"])
        else:
            columns[Column(f"{p}.Targets.W1")] = ak.zip(targets["W1"])
        if self.include_isr_target:
            columns[Column(f"{p}.Targets.ISR")] = ak.zip(targets["ISR"])

        return columns, []

# ============================================================
# Module 2: HDF5 writer
# ============================================================

@define
class SaveSPANetH5(AnalyzerModule):
    """
    Writes SPANetGenMatch's Source/Targets Columns to an HDF5 file per
    chunk. Modeled directly on SaveEventsH5's local-write -> copyFile()
    pattern (same chunk-keyed naming, same temp-file-then-copy mechanics) --
    lifted rather than reinvented, since that part already does what we need.

    Leave `targets_cols` empty for evaluation-only pipelines where
    SPANetGenMatch either isn't run at all, or is run without truth targets
    -- only Source gets written in that case.

    good_event_col : Column, optional
        If given, events where this Column is False are dropped before
        writing -- reproducing what nano_to_h5_V2.py originally did (bad
        events never made it into the H5 at all), applied here rather than
        via a yaml-level SelectOnColumns step. Leave unset to write every
        event SPANetGenMatch saw.
    """

    prefix: str
    source_col: Column
    targets_cols: List[Column] = field(factory=list)
    good_event_col: Column = None
    output_format: str = (
        "{dataset_name}__{sample_name}__{file_id}"
        "__{chunk.event_start}_{chunk.event_stop}.h5"
    )

    def inputs(self, metadata):
        cols = [self.source_col] + list(self.targets_cols)
        if self.good_event_col is not None:
            cols.append(self.good_event_col)
        return cols

    def outputs(self, metadata):
        return []

    def run(self, columns, params):
        import h5py

        file_id = hashlib.md5(
            columns.metadata["chunk"]["file_path"].encode()
        ).hexdigest().upper()
        uid = str(uuid.uuid4())

        target_name = dotFormat(
            self.output_format,
            **dict(dictToDot(columns.metadata)),
            file_id=file_id,
            uuid=uid,
        )
        target = f"{self.prefix}/{target_name}"

        base = Path("localsaved")
        base.mkdir(exist_ok=True, parents=True)
        local_filename = base / f"{uid}.h5"

        if self.good_event_col is not None:
            good_np = ak.to_numpy(columns[self.good_event_col])
        else:
            good_np = None

        try:
            source = columns[self.source_col]
            with h5py.File(local_filename, "w") as f:
                for field_name in source.fields:
                    data = ak.to_numpy(source[field_name])
                    if good_np is not None:
                        data = data[good_np]
                    f.create_dataset(
                        f"INPUTS/Source/{field_name}",
                        data=data,
                        compression="gzip",
                    )
                for tcol in self.targets_cols:
                    tdata = columns[tcol]
                    # e.g. Column("SPANet.Targets.H2") -> HDF5 group "H2"
                    group_name = str(tcol).split(".")[-1]
                    for field_name in tdata.fields:
                        data = ak.to_numpy(tdata[field_name])
                        if good_np is not None:
                            data = data[good_np]
                        f.create_dataset(
                            f"TARGETS/{group_name}/{field_name}",
                            data=data,
                            compression="gzip",
                        )
            copyFile(local_filename, target)
        finally:
            local_filename.unlink(missing_ok=True)

        return columns, []
