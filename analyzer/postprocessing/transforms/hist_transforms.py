from __future__ import annotations

import copy
import itertools as it
from analyzer.postprocessing.style import Style
from analyzer.core.results import Histogram
import numpy as np
from analyzer.utils.structure_tools import dictToDot, dotFormat
import hist
from collections import ChainMap, OrderedDict
from analyzer.utils.querying import BasePattern
from analyzer.utils.structure_tools import (
    ItemWithMeta,
)
from attrs import define, field, asdict
from .registry import TransformHistogram


@define
class SelectAxesValues(TransformHistogram):
    select_axes_values: dict[str, list[str] | list[int] | list[float]]

    def __call__(self, items: list[ItemWithMeta]):
        ret = []
        for item, meta in items:
            h = item.histogram
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
class MergeAxes(TransformHistogram):
    merge_axis_names: list[str | int]

    def __call__(self, items):
        ret = []
        for item, meta in items:
            h = item.histogram
            merging = {x: slice(0, len(h.axes[x]), sum) for x in self.merge_axis_names}
            h = h[merging]
            ret.append(
                ItemWithMeta(Histogram(name=item.name, axes=[], histogram=h), meta)
            )
        return ret


@define
class SplitAxes(TransformHistogram):
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
class NormalizeSystematicByProjection(TransformHistogram):
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

            copy.deepcopy(ph.provenance)
            ret.append(
                ItemWithMeta(
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), metadata=meta
                )
            )

        return ret


@define
class OrBinaryAxes(TransformHistogram):
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
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), metadata=meta
                )
            )

        return ret


@define
class RebinAxes(TransformHistogram):
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
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), metadata=meta
                )
            )

        return ret


@define
class SliceAxes(TransformHistogram):
    slices: dict[str | int, tuple[int | float | None, int | float | None]]

    def __call__(self, items):
        ret = []
        for ph, meta in items:
            h = ph.histogram
            slices = {
                x: slice(*(hist.loc(y) if y else y for y in z))
                for x, z in self.slices.items()
            }
            h = h[slices]
            meta = ChainMap(
                meta,
                {"axis_params": ChainMap(meta.get("axis_params", {}), slices)},
            )
            ret.append(
                ItemWithMeta(
                    Histogram(name=ph.name, axes=ph.axes, histogram=h), metadata=meta
                )
            )

        return ret


@define
class MultiSliceAxes(TransformHistogram):
    multi_slices: dict[str | int, tuple[float, float, int]]

    def __call__(self, items):
        ret = []

        def makePairs(l):
            return np.stack([l[:-1], l[1:]], axis=1)

        for ph, meta in items:
            h = ph.histogram
            chopping = OrderedDict(chopping)
            names, vals = list(chopping.keys()), list(chopping.values())
            pairs = [makePairs(np.arange(*x)) for x in vals]
            for ranges in it.combinations(*pairs):
                slices = {
                    x: slice(*(hist.loc(y) for y in z)) for x, z in zip(names, ranges)
                }
                hnew = h[slices]
                meta = ChainMap(
                    meta,
                    {"axis_params": ChainMap(meta.get("axis_params", {}), slices)},
                )
                ret.append(
                    ItemWithMeta(
                        Histogram(name=ph.name, axes=ph.axes, histogram=hnew),
                        metadata=meta,
                    )
                )

        return ret


@define
class FormatTitle(TransformHistogram):
    title_format: str

    def __call__(self, histograms):
        ret = []
        for ph, meta in histograms:
            meta = ChainMap(
                meta,
                {"title": dotFormat(self.title_format, **dict(dictToDot(meta)))},
            )
            ret.append(ItemWithMeta(ph, metadata=meta))
        return ret


@define
class SetStyle(TransformHistogram):
    style: Style

    def __call__(self, histograms):
        ret = []
        for ph, meta in histograms:
            meta = ChainMap(meta, {"style": asdict(self.style)})
            ret.append(ItemWithMeta(ph, metadata=meta))
        return ret


@define
class ABCDTransformer(TransformHistogram):
    csv_path: str
    x_axis_name: str
    y_axis_name: str
    target_axis_name: str
    key_format: str
    start_idx: int = 1
    _edges: dict[str, tuple[float, float]] = field(init=False, factory=dict)

    """
    Transforms a 2D histogram into a 1D categorical histogram with 4 bins (A, B, C, D) based on x and y cuts.

    The quadrants are defined as:
    - A: x < x_edge, y >= y_edge
    - B: x >= x_edge, y >= y_edge
    - C: x < x_edge, y < y_edge
    - D: x >= x_edge, y < y_edge

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing the cut values. The CSV should have columns:
        key_column, x_edge, y_edge
    x_axis_name : str
        Name of the x-axis in the histogram.
    y_axis_name : str
        Name of the y-axis in the histogram.
    target_axis_name : str
        Name of the new categorical axis to be created.
    key_format : str
        Format string to construct the lookup key from histogram metadata (e.g. "{dataset_name}_{era.name}").
    start_idx : int, optional
        Row index to start reading the CSV file (default is 1 to skip header).
    """

    def __attrs_post_init__(self):
        import csv

        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            for row in rows[self.start_idx :]:
                key = row[0].strip()
                x = float(row[1])
                y = float(row[2])
                self._edges[key] = (x, y)

    def __call__(self, items):
        ret = []
        for ph, meta in items:
            h = ph.histogram

            formatted_key = dotFormat(self.key_format, **dict(dictToDot(meta)))

            x_edge, y_edge = self._edges[formatted_key]

            def makeSlice(x_slice, y_slice):
                return {
                    self.x_axis_name: slice(*x_slice, sum),
                    self.y_axis_name: slice(*y_slice, sum),
                }

            s_a = makeSlice((None, x_edge), (y_edge, None))
            s_b = makeSlice((x_edge, None), (y_edge, None))
            s_c = makeSlice((None, x_edge), (None, y_edge))
            s_d = makeSlice((x_edge, None), (None, y_edge))

            h_a, h_b, h_c, h_d = h[s_a], h[s_b], h[s_c], h[s_d]
            cat_axis = hist.axis.StrCategory(
                ["A", "B", "C", "D"], name=self.target_axis_name, label="Region"
            )
            new_axes = list(h_a.axes) + [cat_axis]
            nh = hist.Hist(*new_axes, storage=h.storage_type())

            v_a, v_b, v_c, v_d = (
                h_a.view(flow=True),
                h_b.view(flow=True),
                h_c.view(flow=True),
                h_d.view(flow=True),
            )

            stacked_view = np.stack([v_a, v_b, v_c, v_d], axis=-1)
            nh.view(flow=True)[...] = stacked_view
            ret.append(
                ItemWithMeta(
                    Histogram(name=ph.name, axes=None, histogram=nh), metadata=meta
                )
            )

        return ret
