import copy
import functools as ft
import itertools as it
import string
from dataclasses import dataclass, field
from typing import Any, ClassVar

from analyzer.core.results import SectorResult
from analyzer.core.specifiers import SectorParams
from pydantic import AfterValidator, BaseModel, Field, field_validator

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
    title_format: str = "{title}"

    @field_validator("axis_options", mode="after")
    @classmethod
    def coerceMode(cls, value):
        for k in list(value.keys()):
            if value[k] in ("Sum", "Split"):
                value[k] = Mode[value[k]]
        return value

    @field_validator("cat_remap", mode="before")
    @classmethod
    def transformCatRemap(cls, value):
        return {(k1, k2): v for k2, v in inner for k1, inner in value.items()}


class PackagedHist(BaseModel):
    histogram: Any
    title: str
    sector_parameters: SectorParams
    axis_parameters: dict[str, Any] | None = None

    @property
    def dim(self):
        return len(self.histogram.axes)

    @property
    def all_parameters(self):
        return {**self.parameters, **(self.all_parameters or {})}

    def compatible(self, other):
        return self.histogram.axes == other.histogram.axes


@dataclass
class SectorGroup:
    separator: ClassVar[str] = " "
    parameters: dict[Any, Any]
    sectors: list[SectorResult]
    axis_options: dict[str, Mode] | None = None
    title_format: str = "{title}"

    cat_remap: dict[tuple[str, int | str], str] | None = None

    @property
    def all_parameters(self):
        return {**self.parameters, **(self.axis_options or {})}

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

        return self.title_format.format(
            title=sector.sector_params.dataset.title, **cat_values
        )

    def histograms(self, hist_name):
        everything = []
        for sector in self.sectors:
            print(self.axis_options)
            hists, labels = splitHistogram(
                sector.result.histograms[hist_name].histogram,
                self.axis_options or None,
                return_labels=True,
            )
            if isinstance(hists, dict):
                for c, h in hists.items():
                    everything.append(
                        PackagedHist(
                            histogram=h,
                            title=self.__getHistTitle(h, sector, dict(zip(labels, c))),
                            sector_parameters=sector.sector_params,
                            axis_options=self.axis_options,
                        )
                    )
            else:
                everything.append(
                    PackagedHist(
                        histogram=hists,
                        title=self.__getHistTitle(hists, sector),
                        sector_parameters=sector.sector_params,
                        axis_options=self.axis_options,
                    )
                )

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
            title_format=spec.title_format,
            cat_remap=spec.cat_remap,
        )
        for params, sectors in grouped
    ]
