from analyzer.core.analysis_modules import AnalyzerModule
import re
import numba
from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import flatten
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import numpy as np
import itertools as it
from attrs import define, field, evolve
import enum

import logging


from analyzer.core.analysis_modules import (
    MetadataExpr,
    MetadataAnd,
    IsRun,
    IsSampleType,
)


logger = logging.getLogger("analyzer.modules")

@define
class GenPartMinDRMaker(AnalyzerModule):
    """
    Compute the minimum delta R among all unique pairs of gen quarks.
    
    Parameters
    ----------
    input_col : Column
        Column containing the GenPart collection (must have eta, phi fields).
    output_col : Column
        Column where the per-event minimum delta R scalar will be stored.
    """
    input_col: Column
    output_col: Column

    def run(self, columns, params):
        gen_parts = columns[self.input_col]
        pairs = ak.combinations(gen_parts, 2, axis=1)
        qi, qj = ak.unzip(pairs)
        dr_all = qi.delta_r(qj)
        min_dr = ak.min(dr_all, axis=1)
        # Replace None (from empty events) with -1 so histogram can handle it
        columns[self.output_col] = ak.fill_none(min_dr, -1.0)
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
