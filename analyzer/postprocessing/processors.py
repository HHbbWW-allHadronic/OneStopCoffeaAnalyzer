from __future__ import annotations

import abc
from cattrs.strategies import (
    include_subclasses,
    configure_tagged_union,
    configure_union_passthrough,
)
from .transforms.registry import Transform

import functools as ft
from .plots.common import PlotConfiguration
from .style import StyleSet
from analyzer.utils.structure_tools import (
    ItemWithMeta,
)
from .grouping import GroupBuilder
from analyzer.utils.structure_tools import globWithMeta
from attrs import define, field

ResultSet = list[list[ItemWithMeta]]

type PostprocessingGroup = (
    dict[str, PostprocessingGroup] | list[PostprocessingGroup] | list[ItemWithMeta]
)


@define
class BasePostprocessor(abc.ABC):
    inputs: list[list[tuple[str, ...]]]
    structure: GroupBuilder
    style_set: StyleSet | None = field(default=None, kw_only=True)
    plot_configuration: PlotConfiguration | None = field(default=None, kw_only=True)

    def run(self, data, prefix=None):
        for i in self.inputs:
            items = [y for x in (globWithMeta(data, l) for l in i) for y in x]
            for x in self.structure.apply(items):
                yield from self.getRunFuncs(x, prefix)

    @abc.abstractmethod
    def getRunFuncs(self, group: PostprocessingGroup, prefix=None):
        pass


def configureConverter(conv):
    union_strategy = ft.partial(configure_tagged_union, tag_name="name")
    include_subclasses(BasePostprocessor, conv, union_strategy=union_strategy)

    union_strategy = ft.partial(configure_tagged_union, tag_name="name")
    include_subclasses(Transform, conv, union_strategy=union_strategy)

    base_hook = conv.get_structure_hook(BasePostprocessor)
    conv.register_structure_hook(int | float | None, lambda x, t: x)
    configure_union_passthrough(int | float | None, conv)

    def convStr(x):
        if isinstance(x, str):
            return x.split("/")
        return x

    @conv.register_structure_hook
    def _(data, t) -> BasePostprocessor:
        real_inputs = []
        for x in data["inputs"]:
            if isinstance(x, list):
                real_inputs.append([convStr(y) for y in x])
            else:
                real_inputs.append([convStr(x)])
        data["inputs"] = real_inputs
        return base_hook(data, t)
