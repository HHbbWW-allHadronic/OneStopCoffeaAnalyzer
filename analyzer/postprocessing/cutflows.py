from __future__ import annotations

import functools as ft
from pathlib import Path
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
from .plots.plots_1d import plotOne, plotRatio, plotDictAsBars
from .plots.plots_2d import plot2D
from .grouping import GroupBuilder
from analyzer.utils.structure_tools import globWithMeta
from attrs import define, field
import abc


@define
class PlotSelectionFlow(BasePostprocessor):
    output_name: str
    style_set: str | StyleSet = field(factory=StyleSet)
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = doFormatting(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        getter = lambda x: getattr(x, "cutflow")

        yield ft.partial(
            plotDictAsBars,
            group,
            common_meta,
            output_path,
            getter=getter,
            style_set=self.style_set,
            normalize=self.normalize,
            plot_configuration=pc,
        )
