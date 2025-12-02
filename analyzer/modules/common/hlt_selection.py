from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
import operator as op
import awkward as ak
import itertools as it
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
from analyzer.core.columns import addSelection
import copy
from rich import print
import numpy as np
import functools as ft


import awkward as ak
import correctionlib
import pydantic as pyd
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import correctionlib.schemav2 as cs
from functools import lru_cache


@define
class SimpleHLT(AnalyzerModule):
    triggers: list[str]
    selection_name: str = "PassHLT"

    def run(self, columns, params):
        metadata = columns.metadata
        trigger_names = metadata["era"]["trigger_names"]
        hlt = columns["HLT"]
        pass_trigger = ft.reduce(
            op.and_, (hlt[trigger_names[name]] for name in self.triggers)
        )
        addSelection(columns, self.selection_name, pass_trigger)
        return columns, []

    def inputs(self, metadata):
        return [Column(("HLT"))]

    def outputs(self, metadata):
        return [Column(f"Selection.{self.selection_name}")]
