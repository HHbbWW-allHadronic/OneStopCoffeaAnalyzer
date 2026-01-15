from __future__ import annotations
from rich import print

import functools as ft
from typing import Literal
import itertools as it
from .plots.common import PlotConfiguration
from .style import StyleSet
from analyzer.utils.querying import BasePattern, Pattern, gatherByCapture, NO_MATCH
from analyzer.utils.structure_tools import (
    deepWalkMeta,
    SimpleCache,
    ItemWithMeta,
    commonDict,
    dictToDot,
    doFormatting,
)
from .processors import BasePostprocessor
from .plots.plots_1d import plotOne, plotRatio
from .plots.plots_2d import plot2D
from .grouping import GroupBuilder
from analyzer.utils.structure_tools import globWithMeta
from attrs import define, field
import abc

ResultSet = list[list[ItemWithMeta]]


@define
class Histogram1D(BasePostprocessor):
    output_name: str
    style_set: str | StyleSet = field(factory=StyleSet)
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False

    def getRunFuncs(self, group, prefix=None):
        if isinstance(group, dict):
            unstacked = group["unstacked"]
            stacked = group["stacked"]
        else:
            unstacked = group
            stacked = None
        common_meta = commonDict(it.chain((stacked or []), (unstacked or [])))
        output_path = doFormatting(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            plotOne,
            unstacked,
            stacked,
            common_meta,
            output_path,
            style_set=self.style_set,
            normalize=self.normalize,
            plot_configuration=pc,
        )


@define
class RatioPlot(BasePostprocessor):
    output_name: str
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    ratio_ylim: tuple[float, float] = (0, 2)
    ratio_hlines: list[float] = field(factory=lambda: [1.0])
    ratio_height: float = 0.5
    ratio_type: Literal["poisson", "poisson-ratio", "efficiency"] = "poisson"
    no_stack: bool = False

    def getRunFuncs(self, group, prefix=None):
        numerator = group["numerator"]
        denominator = group["denominator"]
        common_meta = commonDict(it.chain(numerator, denominator))
        output_path = doFormatting(
            self.output_name, prefix=prefix, **dict(dictToDot(common_meta))
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            plotRatio,
            denominator,
            numerator,
            output_path,
            self.style_set,
            normalize=self.normalize,
            ratio_ylim=self.ratio_ylim,
            ratio_type=self.ratio_type,
            scale=self.scale,
            ratio_hlines=self.ratio_hlines,
            ratio_height=self.ratio_height,
            no_stack=self.no_stack,
            plot_configuration=pc,
        )


@define
class Histogram2D(BasePostprocessor):
    output_name: str
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False

    def getRunFuncs(self, group, prefix=None):
        if len(group) != 1:
            raise RuntimeError()
        hist = group[0]
        common_meta = commonDict(group)
        output_path = doFormatting(
            self.output_name, prefix=prefix, **dict(dictToDot(common_meta))
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            plot2D,
            hist,
            output_path,
            style_set=self.style_set,
            normalize=self.normalize,
            plot_configuration=self.plot_configuration,
            color_scale=self.scale,
        )
