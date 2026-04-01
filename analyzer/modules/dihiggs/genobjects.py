from analyzer.core.analysis_modules import AnalyzerModule
import re

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import flatten
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import itertools as it
from attrs import define, field, evolve
from ..common.axis import RegularAxis
from ..common.histogram_builder import makeHistogram
from ..common.electrons import CutBasedWPs, cut_mapping as electron_cut_mapping
from ..common.muons import IdWps, IsoWps, cut_mapping as muon_cut_mapping
import enum
import numpy as np

import vector

import correctionlib
import logging


from analyzer.core.analysis_modules import (
    MetadataExpr,
    MetadataAnd,
    IsRun,
    IsSampleType,
)


logger = logging.getLogger("analyzer.modules")

@define
class GenPartFilter(AnalyzerModule):
    """
    This analyzer creates a column from GenPart that has a specific
    pdgId and status code.
    Parameters
    ----------
    input_col: Column
        Column where GenPart objects are located, to be filtered.
    output_col: Column
        Column where promoted items will be stored.
    pdgId: int or list of int
        pdgId(s) of target particle(s). Can be a single int or a list
        of ints to select multiple particle types simultaneously.
    status_flag: int or list of int
        Generator status flag bit(s) for filtering. Can be a single int
        or a list of ints. All flags must be set (AND logic).
    exclude_pdgId: bool, optional
        If True, exclude particles with the given pdgId instead of selecting them.
        By default False.
    require_ancestor_pdgId: int or None, optional
        If set, only keep particles that have a mother or grandmother with
        this pdgId in the decay chain. Checked vectorially up to two levels.
        By default None (no ancestor requirement).
    Notes
    -----
    """
    input_col: Column
    output_col: Column
    pdgId: int | list
    status_flag: int | list
    exclude_pdgId: bool = False
    require_ancestor_pdgId: int | None = None

    def run(self, columns, params):
        genpart = columns[self.input_col]

        # Build pdgId mask
        pdgIds = [self.pdgId] if isinstance(self.pdgId, int) else self.pdgId
        pass_pdgId = ak.zeros_like(genpart.pdgId, dtype=bool)
        for pid in pdgIds:
            pass_pdgId = pass_pdgId | (abs(genpart.pdgId) == pid)
        if self.exclude_pdgId:
            pass_pdgId = ~pass_pdgId

        # Build status flag mask - all flags must be set (AND logic)
        status_flags = [self.status_flag] if isinstance(self.status_flag, int) else self.status_flag
        pass_status_flag = ak.ones_like(genpart.pdgId, dtype=bool)
        for flag in status_flags:
            pass_status_flag = pass_status_flag & ((genpart.statusFlags >> flag) & 1 == 1)

        pass_all = pass_pdgId & pass_status_flag

        # Optionally require a specific ancestor (vectorized, checks up to 2 levels)
        if self.require_ancestor_pdgId is not None:
            ancestor_pid = self.require_ancestor_pdgId
            n_per_event = ak.num(genpart, axis=1)

            # Mother level
            q_mother_idx = genpart.genPartIdxMother
            valid_mother = (q_mother_idx >= 0) & (q_mother_idx < n_per_event)
            safe_mother_idx = ak.where(valid_mother, q_mother_idx, 0)
            mother_pid = abs(genpart.pdgId[safe_mother_idx])

            # Grandmother level
            grandmother_idx = genpart.genPartIdxMother[safe_mother_idx]
            valid_grandmother = (grandmother_idx >= 0) & (grandmother_idx < n_per_event)
            safe_grandmother_idx = ak.where(valid_grandmother, grandmother_idx, 0)
            grandmother_pid = abs(genpart.pdgId[safe_grandmother_idx])

            from_ancestor = (
                (valid_mother & (mother_pid == ancestor_pid)) |
                (valid_grandmother & (grandmother_pid == ancestor_pid))
            )
            pass_all = pass_all & from_ancestor

        columns[self.output_col] = genpart[pass_all]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

@define
class GenDiparticleReconstructor(AnalyzerModule):
    """
    Reconstruct a parent particle's 4-vector by summing the 4-vectors
    of its decay products, filtered by PID from a pre-filtered GenPart collection.

    Parameters
    ----------
    input_col : Column
        Column containing the pre-filtered GenPart collection (e.g. from GenPartDecayWalker).
    output_col : Column
        Column where the reconstructed 4-vector record will be stored.
    daughter_pids : list of int
        List of absolute PDG IDs to select as daughters.
        e.g. [5] for b quarks, [1, 2, 3, 4] for light quarks.
    """
    input_col: Column
    output_col: Column
    daughter_pids: list

    def run(self, columns, params):
        gen_parts = columns[self.input_col]

        # Filter to requested daughter PIDs using absolute value
        pid_mask = ak.zeros_like(gen_parts.pdgId, dtype=bool)
        for pid in self.daughter_pids:
            pid_mask = pid_mask | (abs(gen_parts.pdgId) == pid)
        daughters = gen_parts[pid_mask]

        # Build 4-vectors and sum across daughters per event
        vec = vector.zip({
            "pt":   daughters.pt,
            "eta":  daughters.eta,
            "phi":  daughters.phi,
            "mass": daughters.mass,
        })
        reconstructed = ak.sum(vec, axis=1)

        # Store as a single record collection so FourVecHistograms can consume it
        columns[self.output_col] = ak.zip({
            "pt":   reconstructed.pt,
            "eta":  reconstructed.eta,
            "phi":  reconstructed.phi,
            "mass": reconstructed.mass,
        })
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

@define
class GenBJetMatcher(AnalyzerModule):
    """
    Match GenJets to GenPart b quarks using delta R matching.
    For each GenJet, find the minimum delta R to any b quark.
    If that minimum delta R is below the threshold, the GenJet
    is considered b-matched.

    Parameters
    ----------
    genpart_col : Column
        Column containing the GenPart b quark collection
        (e.g. output of GenPartFilter with pdgId=5).
    genjet_col : Column
        Column containing the GenJet collection.
    output_col : Column
        Column where the b-matched GenJets will be stored.
    dr_threshold : float, optional
        Maximum delta R for a GenJet to be considered matched
        to a b quark. By default 0.4.
    """
    genpart_col: Column
    genjet_col: Column
    output_col: Column
    dr_threshold: float = 0.4

    def run(self, columns, params):
        b_quarks = columns[self.genpart_col]
        gen_jets = columns[self.genjet_col]

        # Broadcast jets and quarks against each other without cartesian
        # jets[:, :, np.newaxis] gives shape (events, jets, 1)
        # b_quarks[:, np.newaxis, :] gives shape (events, 1, quarks)
        # delta_r broadcasts to shape (events, jets, quarks)
        dr = gen_jets[:, :, np.newaxis].delta_r(b_quarks[:, np.newaxis, :])

        # For each jet, find minimum dR to any b quark
        min_dr_per_jet = ak.min(dr, axis=2)

        # Fill None for empty events
        min_dr_per_jet = ak.fill_none(min_dr_per_jet, self.dr_threshold + 1.0)

        # Boolean mask applied directly — preserves event structure
        pass_match = min_dr_per_jet < self.dr_threshold

        columns[self.output_col] = gen_jets[pass_match]
        return columns, []

    def inputs(self, metadata):
        return [self.genpart_col, self.genjet_col]

    def outputs(self, metadata):
        return [self.output_col]

@define
class GenWOrganizer(AnalyzerModule):
    """
    Organizes the two gen-level W bosons per event into on-shell and off-shell
    collections based on proximity to the W pole mass.
    Parameters
    ----------
    input_col : Column
        Column containing the GenPart W boson collection (pdgId=24).
    onshell_col : Column
        Column where the on-shell W boson will be stored.
    offshell_col : Column
        Column where the off-shell W boson will be stored.
    w_mass : float, optional
        W pole mass in GeV. Default is 80.4.
    """
    input_col: Column
    onshell_col: Column
    offshell_col: Column
    w_mass: float = 80.4

    def run(self, columns, params):
        ws = columns[self.input_col]
        delta_mass = abs(ws.mass - self.w_mass)
        onshell_idx = ak.argmin(delta_mass, axis=1, keepdims=True)
        offshell_idx = ak.argmax(delta_mass, axis=1, keepdims=True)
        columns[self.onshell_col] = ws[onshell_idx]
        columns[self.offshell_col] = ws[offshell_idx]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.onshell_col, self.offshell_col]

@define
class GenWQuarkMatcher(AnalyzerModule):
    """
    For each W boson, find the two quarks whose mother is that W
    and compute their pairwise delta R.
    Parameters
    ----------
    w_col : Column
        Column containing a single W boson per event (on-shell or off-shell).
    genpart_col : Column
        Column containing the full GenPart collection for index lookup.
    quark_col : Column
        Column containing the 4 light quarks (GenPart_4q).
    output_col : Column
        Column where the pairwise delta R between matched quarks will be stored.
    """
    w_col: Column
    genpart_col: Column
    quark_col: Column
    output_col: Column

    def run(self, columns, params):
        ws = columns[self.w_col]
        genpart = columns[self.genpart_col]
        quarks = columns[self.quark_col]

        # Get GenPart indices of the W bosons
        w_mask = (abs(genpart.pdgId) == 24) & ((genpart.statusFlags >> 13) & 1 == 1)
        all_w_indices = ak.local_index(genpart, axis=1)[w_mask]
        ws_all = genpart[w_mask]

        # Find local index of this W within Gen_Ws by pt matching
        local_w_idx = ak.argmin(abs(ws_all.pt - ws[:, 0].pt), axis=1)

        # Get the GenPart index of this W
        w_genidx = ak.firsts(all_w_indices[ak.from_regular(local_w_idx[:, np.newaxis])])
        w_genidx = ak.fill_none(w_genidx, -1)
        w_genidx = ak.values_astype(w_genidx, np.int32)

        # Match quarks by mother index
        q_mother = ak.values_astype(quarks.genPartIdxMother, np.int32)
        from_this_w = q_mother == w_genidx
        matched_quarks = quarks[from_this_w]

        # Compute pairwise dR only for events with exactly 2 matched quarks
        has_two = ak.num(matched_quarks, axis=1) >= 2
        valid_matched = matched_quarks[has_two]
        dr_qq_valid = ak.to_numpy(valid_matched[:, 0].delta_r(valid_matched[:, 1]))

        result = np.full(len(matched_quarks), -1.0)
        result[ak.to_numpy(has_two)] = dr_qq_valid
        columns[self.output_col] = ak.Array(result)
        return columns, []

    def inputs(self, metadata):
        return [self.w_col, self.genpart_col, self.quark_col]

    def outputs(self, metadata):
        return [self.output_col]

@define
class GenQuarkPairDRTable(AnalyzerModule):
    """
    Computes delta R for all pairs of gen quarks and counts which pair 
    gives the minimum OR maximum delta R most frequently across all events.
    Can operate on all 6 quark pairs (with b quarks) or just the 6 pairs
    from the 4 light quarks (same-W and cross-W).
    Parameters
    ----------
    q_col : Column
        Column containing the 4 light quarks (GenPart_4q).
    w_col : Column
        Column containing the W bosons (Gen_Ws).
    genpart_col : Column
        Column containing the full GenPart collection.
    output_col : Column
        Column where the per-event pair index will be stored.
    b_col : Column or None, optional
        Column containing the 2 b quarks. If None, only 4q pairs are computed.
    mode : str, optional
        Either 'min' or 'max'. Default is 'min'.
    w_mass : float, optional
        W pole mass in GeV. Default is 80.4.
    """
    q_col: Column
    w_col: Column
    genpart_col: Column
    output_col: Column
    output_col_dr: Column
    b_col: Column | None = None
    mode: str = "min"
    w_mass: float = 80.4

    def run(self, columns, params):
        quarks = columns[self.q_col]
        ws = columns[self.w_col]
        genpart = columns[self.genpart_col]

        # Get GenPart indices of the W bosons
        w_mask = (abs(genpart.pdgId) == 24) & ((genpart.statusFlags >> 13) & 1 == 1)
        w_indices = ak.local_index(genpart, axis=1)[w_mask]

        # Determine which W is on-shell vs off-shell by mass
        delta_mass = abs(ws.mass - self.w_mass)
        onshell_w_local = ak.argmin(delta_mass, axis=1)
        offshell_w_local = ak.argmax(delta_mass, axis=1)

        # Get GenPart index of each W
        onshell_w_genidx = ak.fill_none(
            ak.firsts(w_indices[ak.from_regular(onshell_w_local[:, np.newaxis])]), -1)
        offshell_w_genidx = ak.fill_none(
            ak.firsts(w_indices[ak.from_regular(offshell_w_local[:, np.newaxis])]), -1)
        onshell_w_genidx = ak.values_astype(onshell_w_genidx, np.int32)
        offshell_w_genidx = ak.values_astype(offshell_w_genidx, np.int32)

        # Assign light quarks to on-shell or off-shell W via mother index
        q_mother = ak.values_astype(quarks.genPartIdxMother, np.int32)
        on_quarks = quarks[q_mother == onshell_w_genidx]
        off_quarks = quarks[q_mother == offshell_w_genidx]

        # Validity check
        has_valid = (
            (ak.num(on_quarks, axis=1) >= 2) &
            (ak.num(off_quarks, axis=1) >= 2)
        )

        on_sorted = on_quarks[has_valid][ak.argsort(on_quarks[has_valid].pt, axis=1, ascending=False)]
        off_sorted = off_quarks[has_valid][ak.argsort(off_quarks[has_valid].pt, axis=1, ascending=False)]

        q1 = on_sorted[:, 0]
        q2 = on_sorted[:, 1]
        q3 = off_sorted[:, 0]
        q4 = off_sorted[:, 1]

        if self.b_col is not None:
            b_quarks = columns[self.b_col]
            b_sorted = b_quarks[has_valid][ak.argsort(b_quarks[has_valid].pt, axis=1, ascending=False)]
            has_valid = has_valid & (ak.num(b_quarks, axis=1) >= 2)
            b1 = b_sorted[:, 0]
            b2 = b_sorted[:, 1]
            pairs = [
                (b1, b2),
                (b1, q1), (b1, q2), (b1, q3), (b1, q4),
                (b2, q1), (b2, q2), (b2, q3), (b2, q4),
                (q1, q2),
                (q1, q3), (q1, q4),
                (q2, q3), (q2, q4),
                (q3, q4),
            ]
        else:
            pairs = [
                (q1, q2), 
                (q3, q4),
                (q1, q3), 
                (q1, q4),
                (q2, q3), 
                (q2, q4),
            ]

        dr_values = np.stack([
            ak.to_numpy(p[0].delta_r(p[1]))
            for p in pairs
        ], axis=1)

        if self.mode == "min":
            pair_idx = np.argmin(dr_values, axis=1)
        elif self.mode == "max":
            pair_idx = np.argmax(dr_values, axis=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'min' or 'max'.")

        # Extract the actual dR value for the winning pair
        pair_dr_value = dr_values[np.arange(len(dr_values)), pair_idx]

        result_idx = np.full(len(quarks), -1, dtype=np.int64)
        result_dr = np.full(len(quarks), -1.0, dtype=np.float64)
        result_idx[ak.to_numpy(has_valid)] = pair_idx
        result_dr[ak.to_numpy(has_valid)] = pair_dr_value

        columns[self.output_col] = ak.Array(result_idx)
        columns[self.output_col_dr] = ak.Array(result_dr)
        return columns, []

    def inputs(self, metadata):
        inputs = [self.q_col, self.w_col, self.genpart_col]
        if self.b_col is not None:
            inputs.append(self.b_col)
        return inputs

    def outputs(self, metadata):
        return [self.output_col, self.output_col_dr]

@define
class GenQuarkPairDRByIndex(AnalyzerModule):
    """
    For each unique pair index, creates a histogram of the dR value
    for events where that pair gave the min/max dR.
    Parameters
    ----------
    idx_col : Column
        Column containing the per-event pair index.
    dr_col : Column
        Column containing the per-event dR value of the winning pair.
    output_prefix : str
        Prefix for output histogram names.
    n_pairs : int
        Number of pairs (6 for 4q only, 15 for full 6q).
    """
    idx_col: Column
    dr_col: Column
    output_prefix: str
    n_pairs: int = 15

    PAIR_NAMES_15 = [
        "b1b2",
        "b1q1", "b1q2", "b1q3", "b1q4",
        "b2q1", "b2q2", "b2q3", "b2q4",
        "q1q2_same_W_on",
        "q1q3_cross_W", "q1q4_cross_W",
        "q2q3_cross_W", "q2q4_cross_W",
        "q3q4_same_W_off",
    ]

    PAIR_NAMES_6 = [
        "q1q2_same_W_on",
        "q3q4_same_W_off",
        "q1q3_cross_W",
        "q1q4_cross_W",
        "q2q3_cross_W",
        "q2q4_cross_W",
    ]

    def run(self, columns, params):
        idx_vals = ak.to_numpy(columns[self.idx_col])
        dr_vals = ak.to_numpy(columns[self.dr_col])

        pair_names = self.PAIR_NAMES_15 if self.n_pairs == 15 else self.PAIR_NAMES_6

        for i, name in enumerate(pair_names):
            # Mask to events where this pair gave the min/max dR
            mask = idx_vals == i
            dr_for_pair = np.where(mask, dr_vals, -1.0)
            col = Column((f"{self.output_prefix}_{name}",))
            columns[col] = ak.Array(dr_for_pair)

        return columns, []

    def inputs(self, metadata):
        return [self.idx_col, self.dr_col]

    def outputs(self, metadata):
        pair_names = self.PAIR_NAMES_15 if self.n_pairs == 15 else self.PAIR_NAMES_6
        return [
            Column((f"{self.output_prefix}_{name}",))
            for name in pair_names
        ]
