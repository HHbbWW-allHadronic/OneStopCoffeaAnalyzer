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
    status_code: int
        Generator status_code for filtering.
    Notes
    -----
    """

    input_col: Column
    output_col: Column
    pdgId: int
    status_code: int

    def run(self, columns, params):
        metadata = columns.metadata
        genpart = columns[self.input_col]
        pass_pdgId = abs(genpart.pdgId) == self.pdgId
        pass_status_flag = (genpart.statusFlags>>status_code)&1 == 1
        columns[self.output_col] = genpart[pass_pdgId & pass_status_flag]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

