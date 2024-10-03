import functools as ft
import itertools as it
import string
from dataclasses import dataclass
from typing import Any

from analyzer.core import SectorResult


def getNested(d, s):
    parts = s.split(".")

    def getK(di, p):
        return di[p]

    ret = ft.reduce(getK, parts, d)
    return ret


def doFormatting(s, sector_params, **kwargs):
    parsed = string.Formatter().parse(s)
    d = sector_params.model_dump()
    s = ""
    for x in parsed:
        s += x[0]
        if x[1] is not None:
            if x[1] in kwargs:
                s += kwargs[x[1]]
            else:
                s += getNested(d, x[1])
    return s


def groupBy(data, fields):
    def k(v):
        return tuple([getNested(v.sector_params.model_dump(), x) for x in fields])

    grouped = it.groupby(sorted(data, key=k), k)
    ret = [(dict(zip(fields, x)), list(y)) for x, y in grouped]
    return ret


@dataclass
class SectorGroup:
    parameters: dict[str, Any]
    sectors: list[SectorResult]

    def compatible(self, other):
        return self.parameters == other.parameters

    def __len__(self):
        return len(self.sectors)


def createSectorGroups(sectors, *fields):
    grouped = groupBy(sectors, fields)
    return [
        SectorGroup(parameters=params, sectors=sectors) for params, sectors in grouped
    ]
