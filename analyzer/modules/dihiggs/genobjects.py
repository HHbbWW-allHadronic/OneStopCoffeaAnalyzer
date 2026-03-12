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
    pdgId: int
        pdgId of target particle.
    status_flag: int
        Generator status_flag for filtering.
    exclude_pdgId: bool, optional
        If True, exclude particles with the given pdgId instead of selecting them.
        By default False.
    Notes
    -----
    """
    input_col: Column
    output_col: Column
    pdgId: int
    status_flag: int
    exclude_pdgId: bool = False

    def run(self, columns, params):
        metadata = columns.metadata
        genpart = columns[self.input_col]
        if self.exclude_pdgId:
            pass_pdgId = abs(genpart.pdgId) != self.pdgId
        else:
            pass_pdgId = abs(genpart.pdgId) == self.pdgId
        pass_status_flag = (genpart.statusFlags >> self.status_flag) & 1 == 1
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
    For each b quark, find the nearest GenJet. If the minimum delta R
    is below the threshold, the GenJet is considered b-matched.

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

        # For each b quark, compute dR to all GenJets
        # Use ak.cartesian to get all (quark, jet) pairs per event
        pairs = ak.cartesian({"quark": b_quarks, "jet": gen_jets}, axis=1)
        quarks, jets = ak.unzip(pairs)

        # Compute dR for all pairs
        dr = quarks.delta_r(jets)

        # For each b quark, find the index of the nearest GenJet
        nearest_jet_idx = ak.argmin(dr, axis=1)
        min_dr = ak.min(dr, axis=1)

        # Keep only matches within the threshold
        pass_dr = min_dr < self.dr_threshold

        # Get the matched GenJet indices that pass the threshold
        matched_idx = nearest_jet_idx[pass_dr]

        # Collect unique matched GenJet indices per event
        # (avoid duplicate jets if multiple quarks match the same jet)
        matched_jets = gen_jets[matched_idx]

        columns[self.output_col] = matched_jets
        return columns, []

    def inputs(self, metadata):
        return [self.genpart_col, self.genjet_col]

    def outputs(self, metadata):
        return [self.output_col]
