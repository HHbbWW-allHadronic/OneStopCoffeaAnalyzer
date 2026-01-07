from __future__ import annotations
import copy
import itertools as it
import numpy as np
from analyzer.utils.structure_tools import dictToDot, doFormatting
import hist
from rich import print
from cattrs.converters import Converter, BaseConverter
from typing import TypeVar, Generic
from collections import ChainMap, OrderedDict
from analyzer.utils.querying import BasePattern, Pattern, gatherByCapture, NO_MATCH
from analyzer.core.results import Histogram
from analyzer.utils.structure_tools import (
    deepWalkMeta,
    SimpleCache,
    ItemWithMeta,
    commonDict,
)
from analyzer.utils.structure_tools import globWithMeta
from attrs import define
from .transforms.registry import Transform
import abc

ResultSet = list[list[ItemWithMeta]]

T = TypeVar("T")


@define
class GroupBuilder:
    group: BasePattern | None = None
    select: BasePattern | None = None
    subgroups: list[GroupBuilder] | dict[str, GroupBuilder] | None = None
    transforms: list[Transform] | None = None

    def apply(self, items):
        if self.select is not None:
            items = [x for x in items if self.select.match(x.metadata)]
        print(items)
        if not self.group:
            return items
        gathered = gatherByCapture(self.group, items)
        groups: ResultSet = [g.items for g in gathered if g.capture is not NO_MATCH]

        for transform in self.transforms or []:
            groups = [transform(g) for g in groups]

        if self.subgroups is None:
            # if len(groups) == 1:
            #     return groups[0]
            # else:
            return groups

        ret = []
        for group_items in groups:
            if isinstance(self.subgroups, dict):
                r = {}
                for x, y in self.subgroups.items():
                    print(f"IN SUBGROUP {x}")
                    r[x] = y.apply(group_items)
                    print(f"LEAVING SUBGROUP {x}")
            elif isinstance(self.subgroups, list):
                r = []
                for x in self.subgroups:
                    r.append(x.apply(group_items))
            ret.append(r)

        return ret


def configureConverter(conv):
    base_list_str = conv.get_structure_hook(list[str])
    base_list_int = conv.get_structure_hook(list[int])
    base_list_float = conv.get_structure_hook(list[float])

    @conv.register_structure_hook
    def _(data, t) -> list[str] | list[int] | list[float]:
        if len(data) == 0:
            return []
        if isinstance(data, str):
            return [data]
        if isinstance(data[0], str):
            return base_list_str(data, list[str])
        elif isinstance(data[0], int):
            return base_list_str(data, list[int])
        else:
            return base_list_str(data, list[float])

    @conv.register_structure_hook
    def _(data, t) -> list[GroupBuilder] | dict[str, GroupBuilder] | None:
        if data is None:
            return None
        if isinstance(data, list):
            return [conv.structure(x, GroupBuilder) for x in data]
        if isinstance(data, dict):
            return {k: conv.structure(v, GroupBuilder) for k, v in data.items()}
        else:
            raise RuntimeError()
