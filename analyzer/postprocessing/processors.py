from __future__ import annotations

import abc
from cattrs.strategies import include_subclasses, configure_tagged_union
from .transforms.registry import Transform

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
from .plots.plots_1d import plotOne, plotRatio
from .plots.plots_2d import plot2D
from .grouping import GroupBuilder
from analyzer.utils.structure_tools import globWithMeta
from attrs import define, field
import abc

ResultSet = list[list[ItemWithMeta]]

type PostprocessingGroup = (
    dict[str, PostprocessingGroup] | list[PostprocessingGroup] | list[ItemWithMeta]
)


@define
class BasePostprocessor(abc.ABC):
    inputs: list[tuple[str, ...]]
    structure: GroupBuilder
    style_set: StyleSet | None = field(default=None, kw_only=True)
    plot_configuration: PlotConfiguration | None = field(default=None, kw_only=True)

    def run(self, data):
        for i in self.inputs:
            items = globWithMeta(data, i)
            for x in self.structure.apply(items):
                yield from self.getRunFuncs(x)

    @abc.abstractmethod
    def getRunFuncs(self, group: PostprocessingGroup, prefix=None):
        pass


def configureConverter(conv):
    union_strategy = ft.partial(configure_tagged_union, tag_name="name")
    include_subclasses(BasePostprocessor, conv, union_strategy=union_strategy)

    union_strategy = ft.partial(configure_tagged_union, tag_name="name")
    include_subclasses(Transform, conv, union_strategy=union_strategy)

    base_hook = conv.get_structure_hook(BasePostprocessor)

    @conv.register_structure_hook
    def _(data, t) -> BasePostprocessor:
        real_inputs = []
        for x in data["inputs"]:
            if isinstance(x, str):
                real_inputs.append(x.split("/"))
            else:
                real_inputs.append(x)
        data["inputs"] = real_inputs
        return base_hook(data, t)
