from analyzer.core.analysis_modules import AnalyzerModule
import re

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import flatten
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import itertools as it
from attrs import define, field, evolve
from .axis import RegularAxis
from .histogram_builder import makeHistogram


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
class Count(AnalyzerModule):
    input_col: Column
    output_col: Column

    def run(self, columns, params):
        columns[self.output_col] = ak.num(columns[self.input_col], axis=1)
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class PromoteIndex(AnalyzerModule):
    """
    Promote a fixed index of a nested collection to a top-level column.

    This analyzer selects a single element at a given index from each
    entry of a nested (jagged) input collection and stores it as a
    top-level column. It is commonly used to extract leading or
    sub-leading objects (e.g. leading jet, first lepton) from per-event
    collections.

    Parameters
    ----------
    input_col : Column
        Column containing a nested collection (e.g. ``N × M`` objects).
    output_col : Column
        Column where the selected elements will be stored.
    index : int, optional
        Index of the element to promote from each nested collection,
        by default ``0`` (leading element).

    """

    input_col: Column
    output_col: Column
    index: int = 0

    def run(self, columns, params):
        columns[self.output_col] = columns[self.input_col][:, self.index]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class Concatenate(AnalyzerModule):
    input_cols: list[Column]
    output_col: Column

    def run(self, columns, params):
        columns[self.output_col] = ak.concatenate(
            [columns[c] for c in self.input_cols], axis=1
        )
        return columns, []

    def inputs(self, metadata):
        return self.input_cols

    def outputs(self, metadata):
        return [self.output_col]
