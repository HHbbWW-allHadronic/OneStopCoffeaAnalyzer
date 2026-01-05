from __future__ import annotations
import copy
import itertools as it
from collections import ChainMap
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
import abc

ResultSet = list[list[ItemWithMeta]]

@define
class SelectAxesValues:
    select_axes_values: dict[str, list[str] | list[int] | list[float]]

    def __call__(self, items: list[ItemWithMeta]):
        ret = []
        for item,meta in items:
            h = item.histogram
            keys_vals = list(self.select_axes_values.items())
            keys, vals = list(zip(*keys_vals))
            # new_axes = [x for x in item.axes if x.name not in select_axes_values]
            for p in it.product(*vals):
                u = dict(zip(keys, p))
                new_meta = ChainMap(meta, u)
                ret.append(ItemWithMeta(Histogram(name=item.name, axes=[], histogram=h[u]),new_meta))
        return ret
    
@define
class MergeAxes:
    merge_axis_names: list[str | int]

    def __call__(self, items):
        ret = []
        for item,meta in items:
            h = item.histogram
            merging = {x: sum for x in self.merge_axis_names}
            h = h[merging]
            ret.append(ItemWithMeta(Histogram(name=item.name, axes=[], histogram=h), meta))
        return ret


@define
class GroupBuilder:
    group: BasePattern
    select: BasePattern | None = None
    subgroups: list[GroupBuilder] | dict[str, GroupBuilder] | None = None
    transforms: None = None

    def apply(self, items):
        if self.select is not None:
            items = [x for x in items if self.select.match(x.metadata)]
        gathered = gatherByCapture(self.group, items)
        groups:  ResultSet = [g.items for g in gathered if g.capture is not NO_MATCH]

        for transform in self.transforms or []:
            groups = [transform(g) for g in groups]

        if self.subgroups is None:
            if len(groups) == 1:
                return groups[0]
            else:
                return groups

        ret = []
        for group_items in groups:
            if isinstance(self.subgroups, dict):
                r = {}
                for x, y in self.subgroups.items():
                    r[x] = y.apply(group_items)
            elif isinstance(self.subgroups, list):
                r = []
                for x in self.subgroups:
                    r.append(x.apply(group_items))
            ret.append(r)

        return ret

