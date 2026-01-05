from __future__ import annotations
import copy
import itertools as it
import numpy as np
from analyzer.utils.structure_tools import dictToDot, doFormatting
import hist
from rich import print
from cattrs.converters import Converter, BaseConverter
from typing import Type
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
import abc

ResultSet = list[list[ItemWithMeta]]


@define
class SelectAxesValues:
    select_axes_values: dict[str, list[str] | list[int] | list[float]]

    def __call__(self, items: list[ItemWithMeta]):
        ret = []
        for item, meta in items:
            h = item.histogram
            print(h)
            keys_vals = list(self.select_axes_values.items())
            keys, vals = list(zip(*keys_vals))
            # new_axes = [x for x in item.axes if x.name not in select_axes_values]
            for p in it.product(*vals):
                u = dict(zip(keys, p))
                new_meta = ChainMap(meta, u)
                ret.append(
                    ItemWithMeta(
                        Histogram(name=item.name, axes=[], histogram=h[u]), new_meta
                    )
                )
        return ret


@define
class MergeAxes:
    merge_axis_names: list[str | int]

    def __call__(self, items):
        ret = []
        for item, meta in items:
            h = item.histogram
            merging = {x: sum for x in self.merge_axis_names}
            h = h[merging]
            ret.append(
                ItemWithMeta(Histogram(name=item.name, axes=[], histogram=h), meta)
            )
        return ret


@define
class SplitAxes:
    split_axis_names: list[str]
    limit_pattern: BasePattern | None = None

    def __call__(self, items):
        import hist

        ret = []
        for ph, meta in items:
            h = ph.histogram

            split_axes = [h.axes[a] for a in self.split_axis_names]

            def passedPattern(name, val):
                if self.limit_pattern is not None:
                    if isinstance(self.limit_pattern, dict):
                        return self.limit_pattern[name].match(val)
                    else:
                        return self.limit_pattern.match(val)
                return True

            possible_values = OrderedDict(
                {
                    (x.name): [y for y in x if passedPattern(x.name, y)]
                    for x in split_axes
                }
            )
            labels = [(x.name or x.label) for x in split_axes]

            all_hists = {
                x: h[dict(zip(possible_values.keys(), map(hist.loc, x)))]
                for x in it.product(*possible_values.values())
            }
            for values, split_hist in all_hists.items():
                axis_values = dict(zip(labels, values))
                meta = ChainMap(
                    meta,
                    {"axis_params": ChainMap(meta.get("axis_params", {}), axis_values)},
                )
                ret.append(
                    ItemWithMeta(
                        Histogram(name=ph.name, axes=None, histogram=split_hist), meta
                    )
                )

        return ret


@define
class NormalizeSystematicByProjection:
    projection_axes: list[str]
    unweighted_name: str
    other_name: str
    target_name: str
    variation_axis: str = "variation"

    # @field_validator("projection_axes", mode="before")
    # @classmethod
    # def makeList(cls, data):
    #     if isinstance(data, str):
    #         return [data]
    #     return data

    def __call__(self, items):
        import hist

        ret = []
        for ph, meta in items:
            hold = ph.histogram
            h = ph.histogram.copy(deep=True)

            proj_axes = [h.axes[x] for x in self.projection_axes]
            for idxs in it.product(
                *((hist.underflow, *range(len(x)), hist.overflow) for x in proj_axes)
            ):
                # print(i, pa.bin(i))
                d = dict(zip((x.name for x in proj_axes), idxs))
                hu = (
                    hold[
                        {
                            **d,
                            self.variation_axis: self.unweighted_name,
                        }
                    ]
                    .sum()
                    .value
                )
                hv = hold[{**d, self.variation_axis: self.other_name}].sum().value

                if hv == 0:
                    continue

                scale = hu / hv
                vals = scale * h[{**d, self.variation_axis: self.target_name}].values()
                weights = (
                    scale**2
                    * h[{**d, self.variation_axis: self.target_name}].variances()
                )

                h[{**d, self.variation_axis: self.target_name}] = np.stack(
                    [vals, weights], axis=-1
                )

            provenance = copy.deepcopy(ph.provenance)
            ret.append(
                ItemWithMeta(
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), meta=meta
                )
            )

        return ret


@define
class OrBinaryAxes:
    or_axis_names: list[str]

    def __call__(self, items):
        ret = []
        for ph, meta in items:
            h = ph.histogram

            to_add = []
            sor = set(self.or_axis_names)
            for i in range(1, 1 + len(self.or_axis_names)):
                for c in it.combinations(self.or_axis_names, i):
                    d = {x: 1 for x in c} | {x: 0 for x in sor - set(c)}
                    to_add.append(h[d])

            h = sum(to_add)
            axis_values = {x: "OR" for x in self.or_axis_names}
            meta = ChainMap(
                meta,
                {"axis_params": ChainMap(meta.get("axis_params", {}), axis_values)},
            )
            ret.append(
                ItemWithMeta(
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), meta=meta
                )
            )

        return ret


@define
class RebinAxes:
    rebin: int | dict[str, int]

    def __call__(self, items):
        ret = []
        for ph, meta in items:
            h = ph.histogram
            if isinstance(self.rebin, dict):
                rebins = {x: hist.rebin(y) for x, y in self.rebin.items()}
            else:
                rebins = {x.name: hist.rebin(self.rebin) for x in h.axes}
            h = h[rebins]
            provenance = copy.deepcopy(ph.provenance)
            provenance.axis_params.update(rebins)
            meta = ChainMap(
                meta,
                {"axis_params": ChainMap(meta.get("axis_params", {}), rebins)},
            )
            ret.append(
                ItemWithMeta(
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), meta=meta
                )
            )

        return ret


@define
class SliceAxes:
    slices: list[tuple[int | float | None, int | float | None]]

    def __call__(self, items):
        ret = []
        for ph, meta in items:
            h = ph.histogram
            slices = dict(
                (a.name, slice(*(hist.loc(x) if x else x for x in s)))
                for a, s in zip(h.axes, self.slices)
            )
            h = h[slices]
            meta = ChainMap(
                meta,
                {"axis_params": ChainMap(meta.get("axis_params", {}), slices)},
            )
            ret.append(
                ItemWithMeta(
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), meta=meta
                )
            )

        return ret


@define
class FormatTitle:
    title_format: str

    def __call__(self, histograms):
        ret = []
        for ph, meta in histograms:
            meta = ChainMap(
                meta,
                {"title": doFormatting(self.title_format, **dict(dictToDot(meta)))},
            )
            ret.append(ItemWithMeta(ph, meta=meta))
        return ret


Transforms = MergeAxes | SelectAxesValues | SplitAxes


@define
class GroupBuilder:
    group: BasePattern
    select: BasePattern | None = None
    subgroups: list[GroupBuilder] | dict[str, GroupBuilder] | None = None
    transforms: list[Transforms] | None = None

    def apply(self, items):
        if self.select is not None:
            items = [x for x in items if self.select.match(x.metadata)]
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
                    r[x] = y.apply(group_items)
            elif isinstance(self.subgroups, list):
                r = []
                for x in self.subgroups:
                    r.append(x.apply(group_items))
            ret.append(r)

        return ret


def configureConverter(conv: Converter):
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
        if isinstance(data, list):
            return {k: conv.structure(v, GroupBuilder) for k, v in data.values()}
        else:
            raise RuntimeError()
