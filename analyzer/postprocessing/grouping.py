import functools as ft
import itertools as it
import copy
import string
from dataclasses import dataclass, field
from typing import Any, ClassVar
from pydantic import BaseModel, Field, field_validator, AfterValidator

from analyzer.core.results import SectorResult

from .split_histogram import Mode, splitHistogram


def getNested(d, s):
    parts = s.split(".")

    def getK(di, p):
        return di[p]

    ret = ft.reduce(getK, parts, d)
    return ret


def doFormatting(s, data, **kwargs):
    parsed = string.Formatter().parse(s)
    d = data
    s = ""
    all_data = {**data, **kwargs}
    for x in parsed:
        s += x[0]
        if x[1] is not None:
            s += all_data[x[1]]
    return s


def groupBy(data, fields):
    def k(v):
        return tuple([getNested(v.params_dict, x) for x in fields])

    grouped = it.groupby(sorted(data, key=k), k)
    ret = [(dict(zip(fields, x)), list(y)) for x, y in grouped]
    return ret


class SectorGroupSpec(BaseModel):
    fields: list[str]
    axis_options: dict[str | int, Mode | str | int] | None = None
    cat_remap: dict[tuple[str, int | str], str] | None = None
    label_format: str = "{title}"

    @field_validator("axis_options", mode="after")
    @classmethod
    def coerceMode(cls, value):
        for k in list(value.keys()):
            if value[k] in ("Sum", "Split"):
                value[k] = Mode[value[k]]
        return value


@dataclass
class SectorGroup:
    separator: ClassVar[str] = " "
    parameters: dict[Any, Any]
    sectors: list[SectorResult]
    axis_options: dict[str, Mode] | None = None
    label_format: str = "{title}"

    cat_remap: dict[tuple[str, int | str], str] | None = None

    def compatible(self, other):
        return self.parameters == other.parameters

    def __len__(self):
        return len(self.sectors)

    def __iter__(self):
        return iter(self.sectors)

    def __getHistTitle(self, hist, sector, cat_values=None):
        cat_values = cat_values or {}
        print(cat_values)
        l = copy.deepcopy(cat_values)
        if self.cat_remap:
            for k, v in self.cat_remap.items():
                if k in l:
                    l[k] = v
        
        return self.label_format.format(
            title=sector.sector_params.dataset.title, **cat_values
        )

    def histograms(self, hist_name):
        everything = {}
        for sector in self.sectors:
            print(self.axis_options)
            hists, labels = splitHistogram(
                sector.result.histograms[hist_name].histogram,
                self.axis_options,
                return_labels=True,
            )
            if isinstance(hists, dict):
                for c, h in hists.items():
                    everything[self.__getHistTitle(h, sector, dict(zip(labels, c)))] = h
            else:
                everything[self.__getHistTitle(hists, sector)] = hists

        return everything

    def __str__(self):
        return self.dict(exclude=["sectors"])


def createSectorGroups(sectors, spec):
    grouped = groupBy(sectors, spec.fields)
    return [
        SectorGroup(
            parameters=params,
            sectors=sectors,
            axis_options=spec.axis_options,
            label_format=spec.label_format,
            cat_remap=spec.cat_remap,
        )
        for params, sectors in grouped
    ]
