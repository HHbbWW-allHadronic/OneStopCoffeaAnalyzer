from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import addSelection
import pickle
import warnings
from analyzer.core.columns import Column
import awkward as ak
import itertools as it
from attrs import define, field
from ..common.axis import RegularAxis
from ..common.histogram_builder import makeHistogram
import copy
from rich import print
import numpy as np


import awkward as ak
import correctionlib
import pydantic as pyd
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import correctionlib.schemav2 as cs
from functools import lru_cache


@define
class VecDRSelection(AnalyzerModule):
    input_col: Column
    selection_name: str
    min_dr: float | None = None
    max_dr: float | None = None
    idx_1: int = 0
    idx_2: int = 1

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]

    def run(self, columns, params):
        col = columns[self.input_col]
        filled = ak.pad_none(col, max([self.idx_1, self.idx_2])+1, axis=1)
        dr = ak.fill_none(
            filled[:, self.idx_1].delta_r(filled[:, self.idx_2]), False
        )
        sel =None
        if self.min_dr is not None:
            sel = dr >= self.min_dr
        if self.max_dr is not None:
            if sel is not None:
                sel = sel & (dr <= self.max_dr)
            else:
                sel = dr <= self.max_dr

        addSelection(columns, self.selection_name, sel)
        return columns, []
