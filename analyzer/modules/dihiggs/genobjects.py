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
    Notes
    -----
    """
    input_col: Column
    output_col: Column
    pdgId: int | list
    status_flag: int | list
    exclude_pdgId: bool = False

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

        columns[self.output_col] = genpart[pass_pdgId & pass_status_flag]
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
    For each W boson, find the two nearest light quarks and compute
    their pairwise delta R.
    Parameters
    ----------
    w_col : Column
        Column containing a single W boson per event (on-shell or off-shell).
    quark_col : Column
        Column containing the 4 light quarks (GenPart_4q).
    output_col : Column
        Column where the pairwise delta R between matched quarks will be stored.
    """
    w_col: Column
    quark_col: Column
    output_col: Column

    def run(self, columns, params):
        ws = columns[self.w_col]
        quarks = columns[self.quark_col]

        dr = ws[:, :, np.newaxis].delta_r(quarks[:, np.newaxis, :])
        dr = dr[:, 0, :]
        nearest_two = ak.argsort(dr, axis=1)[:, :2]
        matched_quarks = quarks[nearest_two]

        dr_qq = matched_quarks[:, 0].delta_r(matched_quarks[:, 1])
        columns[self.output_col] = ak.fill_none(dr_qq, -1.0)
        return columns, []

    def inputs(self, metadata):
        return [self.w_col, self.quark_col]

    def outputs(self, metadata):
        return [self.output_col]

@define
class GenQuarkPairDRTable(AnalyzerModule):
    """
    Computes delta R for all 15 unique pairs of the 6 gen quarks
    (2 b quarks + 2 on-shell W quarks + 2 off-shell W quarks) and
    counts which pair gives the minimum delta R most frequently across
    all events. Quarks are pT-ordered within each category.
    Parameters
    ----------
    b_col : Column
        Column containing the 2 b quarks (GenPart_2b).
    q_col : Column
        Column containing the 4 light quarks (GenPart_4q).
    w_onshell_col : Column
        Column containing the on-shell W boson (Gen_W_onshell).
    w_offshell_col : Column
        Column containing the off-shell W boson (Gen_W_offshell).
    output_col : Column
        Column where the per-event minimum pair index will be stored.
    """
    b_col: Column
    q_col: Column
    w_onshell_col: Column
    w_offshell_col: Column
    output_col: Column

    PAIR_LABELS = [
        "b1-b2",
        "b1-q1", "b1-q2", "b1-q3", "b1-q4",
        "b2-q1", "b2-q2", "b2-q3", "b2-q4",
        "q1-q2",
        "q1-q3", "q1-q4",
        "q2-q3", "q2-q4",
        "q3-q4",
    ]

    def run(self, columns, params):
        b_quarks = columns[self.b_col]
        quarks = columns[self.q_col]
        w_onshell = columns[self.w_onshell_col]
        w_offshell = columns[self.w_offshell_col]

        # pT-order b quarks
        b_sorted = b_quarks[ak.argsort(b_quarks.pt, axis=1, ascending=False)]
        b1 = b_sorted[:, 0]
        b2 = b_sorted[:, 1]

        # Assign light quarks to on-shell W using dR
        dr_on = w_onshell[:, :, np.newaxis].delta_r(quarks[:, np.newaxis, :])
        dr_on = dr_on[:, 0, :]
        nearest_on = ak.argsort(dr_on, axis=1)[:, :2]
        on_quarks = quarks[nearest_on]
        on_sorted = on_quarks[ak.argsort(on_quarks.pt, axis=1, ascending=False)]
        q1 = on_sorted[:, 0]
        q2 = on_sorted[:, 1]

        # Assign light quarks to off-shell W using dR
        dr_off = w_offshell[:, :, np.newaxis].delta_r(quarks[:, np.newaxis, :])
        dr_off = dr_off[:, 0, :]
        nearest_off = ak.argsort(dr_off, axis=1)[:, :2]
        off_quarks = quarks[nearest_off]
        off_sorted = off_quarks[ak.argsort(off_quarks.pt, axis=1, ascending=False)]
        q3 = off_sorted[:, 0]
        q4 = off_sorted[:, 1]

        # Compute dR for all 15 pairs and stack into (events, 15)
        pairs = [
            (b1, b2),
            (b1, q1), (b1, q2), (b1, q3), (b1, q4),
            (b2, q1), (b2, q2), (b2, q3), (b2, q4),
            (q1, q2),
            (q1, q3), (q1, q4),
            (q2, q3), (q2, q4),
            (q3, q4),
        ]
        dr_values = np.stack([
            ak.to_numpy(p[0].delta_r(p[1]))
            for p in pairs
        ], axis=1)

        min_pair_idx = np.argmin(dr_values, axis=1)
        columns[self.output_col] = ak.Array(min_pair_idx.astype(np.int64))
        return columns, []

    def inputs(self, metadata):
        return [self.b_col, self.q_col, self.w_onshell_col, self.w_offshell_col]

    def outputs(self, metadata):
        return [self.output_col]
