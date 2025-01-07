import functools as ft
import itertools as it
import string
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel ,Field

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


@dataclass
class SectorGroup:
    parameters: dict[Any, Any]
    sectors: list[SectorResult]
    axis_options: dict[str, Mode] | None = None

    def compatible(self, other):
        return self.parameters == other.parameters

    def __len__(self):
        return len(self.sectors)

    def __iter__(self):
        return iter(self.sectors)



class SectorGroupSpec(BaseModel):
    fields: list[str]
    axis_options: dict[str, Mode] | None = None


def createSectorGroups(sectors, *fields, axis_options=None):
    grouped = groupBy(sectors, fields)
    return [
        SectorGroup(parameters=params, sectors=sectors, axis_options=axis_options)
        for params, sectors in grouped
    ]
