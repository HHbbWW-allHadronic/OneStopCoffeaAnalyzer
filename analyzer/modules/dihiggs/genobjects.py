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
    intpu_col: Column
        Column where GenPart objects are located, to be filtered.
    output_col: Column
        Column where promoted items will be stored.
    pdgId: int
        pdgId of target particle.
    status_flag: int
        Generator status_flag for filtering.
    Notes
    -----
    """

    input_col: Column
    output_col: Column
    pdgId: int
    status_flag: int

    def run(self, columns, params):
        metadata = columns.metadata
        genpart = columns[self.input_col]
        pass_pdgId = abs(genpart.pdgId) == self.pdgId
        pass_status_flag = (genpart.statusFlags>>self.status_flag)&1 == 1
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
    output_prefix : str
        Prefix for the output columns. Will produce:
        <output_prefix>_pt, <output_prefix>_eta, <output_prefix>_phi, <output_prefix>_mass
    daughter_pids : list of int
        List of absolute PDG IDs to select as daughters.
        e.g. [5] for b quarks, [1, 2, 3, 4] for light quarks.
    """
    input_col: Column
    output_prefix: str
    daughter_pids: list

    def run(self, columns, params):
        gen_parts = columns[self.input_col]

        # Filter to requested daughter PIDs using absolute value
        pid_mask = ak.zeros_like(gen_parts.pdgId, dtype=bool)
        for pid in self.daughter_pids:
            pid_mask = pid_mask | (abs(gen_parts.pdgId) == pid)

        daughters = gen_parts[pid_mask]

        # Build 4-vectors in Cartesian coordinates for correct summation
        vec = vector.zip({
            "pt":   daughters.pt,
            "eta":  daughters.eta,
            "phi":  daughters.phi,
            "mass": daughters.mass,
        })

        # Sum 4-vectors across daughters per event
        reconstructed = ak.sum(vec, axis=1)

        # Store output as flat scalar columns
        columns[Column((f"{self.output_prefix}_pt",))]   = reconstructed.pt
        columns[Column((f"{self.output_prefix}_eta",))]  = reconstructed.eta
        columns[Column((f"{self.output_prefix}_phi",))]  = reconstructed.phi
        columns[Column((f"{self.output_prefix}_mass",))] = reconstructed.mass

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [
            Column((f"{self.output_prefix}_pt",)),
            Column((f"{self.output_prefix}_eta",)),
            Column((f"{self.output_prefix}_phi",)),
            Column((f"{self.output_prefix}_mass",)),
        ]
