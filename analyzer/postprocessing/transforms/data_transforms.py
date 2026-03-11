from __future__ import annotations
import copy
from analyzer.core.results import Histogram, SavedColumns
import hist
from collections import OrderedDict
from analyzer.utils.structure_tools import (
    ItemWithMeta,
)
from attrs import define
from .registry import TransformSavedColumns
from analyzer.modules.common.axis import Axis


@define
class MaskData(TransformSavedColumns):
    mask: str

    def __call__(self, items: list[ItemWithMeta]):
        ret = []
        for item, meta in items:
            data = item.data
            m = eval(self.mask, None, {**locals(), **data})
            data = {x: y[m] for x, y in data.items()}

            ret.append(
                ItemWithMeta(
                    SavedColumns(name=item.name, data=data),
                    metadata=meta,
                )
            )
        return ret


@define
class AddData(TransformSavedColumns):
    new_col: str
    func: str

    def __call__(self, items: list[ItemWithMeta]):
        ret = []
        for item, meta in items:
            data = copy.copy(item.data)
            m = eval(self.func, None, {**locals(), **data})
            data[self.new_col] = m

            ret.append(
                ItemWithMeta(
                    SavedColumns(name=item.name, data=data),
                    metadata=meta,
                )
            )
        return ret


@define
class MakeHistogram(TransformSavedColumns):
    column_axis_mapping: dict[str, Axis]
    histogram_name: str
    weight_col: str | None = None

    def __call__(self, items: list[ItemWithMeta]):
        ordered = OrderedDict(self.column_axis_mapping)
        axes = [a.toHist() for a in ordered.values()]
        base_h = hist.Hist(
            *axes,
            storage=hist.storage.Double()
            if self.weight_col is None
            else hist.storage.Weight(),
        )
        ret = []
        for item, meta in items:
            h = base_h.copy(deep=True)
            data = item.data

            if self.weight_col is None:
                h.fill(*[data[x] for x in ordered])
            else:
                h.fill(*[data[x] for x in ordered], weight=data[self.weight_col])
            ret.append(
                ItemWithMeta(
                    Histogram(
                        name=self.histogram_name,
                        axes=list(self.column_axis_mapping.values()),
                        histogram=h,
                    ),
                    metadata=meta,
                )
            )
        return ret
