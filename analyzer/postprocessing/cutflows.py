from __future__ import annotations

import functools as ft
from typing import Literal
from .style import StyleSet
from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    doFormatting,
)
from .processors import BasePostprocessor
from .plots.plots_1d import plotDictAsBars
from attrs import define, field


def _get_cutflow(x):
    return getattr(x, "cutflow")


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

        yield ft.partial(
            plotDictAsBars,
            group,
            common_meta,
            output_path,
            getter=_get_cutflow,
            style_set=self.style_set,
            normalize=self.normalize,
            plot_configuration=pc,
        )
