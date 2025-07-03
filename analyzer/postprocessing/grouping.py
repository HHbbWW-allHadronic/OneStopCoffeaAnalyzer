import copy
import operator as op
import functools as ft
import itertools as it
import string
from typing import Annotated, Any, ClassVar

from analyzer.core.results import SectorResult
from analyzer.core.specifiers import SectorParams, SectorSpec
from pydantic import BaseModel, Field, field_validator

from .split_histogram import Mode, splitHistogram
from .style import Style, StyleSet


def getNested(d, s):
    if s in d:
        return d[s]
    parts = s.split(".", 1)
    try:
        ret = ft.reduce(getNested, parts, d)
    except KeyError as e:
        raise KeyError(str(s))
    return ret


def doFormatting(s, data=None, **kwargs):
    parsed = string.Formatter().parse(s)
    d = data or {}
    s = ""
    all_data = {**d, **kwargs}
    for x in parsed:
        s += x[0]
        if x[1] is not None:
            if x[1] in all_data:
                s += str(all_data[x[1]])
            else:
                s += str(getNested(all_data, x[1]))
    return s


def groupBy(data, fields, data_acquire=lambda x: x):
    def k(v):
        return tuple([getNested(data_acquire(v), x) for x in fields])

    grouped = it.groupby(sorted(data, key=k), k)
    ret = [(dict(zip(fields, x)), list(y)) for x, y in grouped]
    return ret


class SectorGroupSpec(BaseModel):
    fields: list[str]
    to_process: SectorSpec | None = None
    axis_options: dict[str | int, Mode | str | int] | None = None
    cat_remap: dict[tuple[str, int | str], str] | None = None
    title_format: str = "{title}"
    style_set: StyleSet | None = None
    add_together: bool = False

    @field_validator("axis_options", mode="after")
    @classmethod
    def coerceMode(cls, value):
        for k in list(value.keys()):
            if value[k] in ("Sum", "Split", "Or", "Rebin2", "Rebin3", "Rebin4"):
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
    style: Style | None = None

    @property
    def dim(self):
        return len(self.histogram.axes)

    @property
    def all_parameters(self):
        return {**self.parameters, **(self.all_parameters or {})}

    def compatible(self, other):
        return self.histogram.axes == other.histogram.axes


class SectorGroupParameters(BaseModel):
    parameters: dict[Any, Any]
    axis_options: dict[str, Mode | str | int] | None = None

    @property
    def all_parameters(self):
        return {**self.parameters, **(self.axis_options or {})}

    def compatible(self, other):
        return self.parameters == other.parameters


def groupsMatch(group1, group2, fields):
    return all(group1.all_parameters[f] == group2.all_parameters[f] for f in fields)


class SectorGroup(SectorGroupParameters):
    """
    A collection of sectors (Region,Datasets pairs), which are treated as a unit for certain processors purposes.
    Different processors may use this construction differently

    """

    separator: ClassVar[str] = " "
    sectors: Annotated[list[SectorResult], Field(repr=False)]
    title_format: Annotated[str, Field(repr=False)] = "{title}"
    style_set: Annotated[StyleSet | None, Field(repr=False)] = None
    cat_remap: dict[tuple[str, int | str], str] | None = None
    add_together: bool = False

    def __len__(self):
        return len(self.sectors)

    def __iter__(self):
        return iter(self.sectors)

    def __getHistTitle(self, hist, sector, cat_values=None):
        cat_values = cat_values or {}
        l = copy.deepcopy(cat_values)
        if self.cat_remap:
            for k, v in self.cat_remap.items():
                if k in l:
                    l[k] = v
        return doFormatting(
            self.title_format,
            **sector.sector_params.model_dump(),
            title=sector.sector_params.dataset.title,
        )

    def getSectorStyle(self, sector):
        if self.style_set:
            style = self.style_set.getStyle(sector.sector_params)
        else:
            return None

    def histograms(self, hist_name):
        """
        Get all the histograms corresponding to the given name.
        Returns a list of PackagedHist
        """
        everything = []
        for sector in self.sectors:
            h = sector.result.histograms[hist_name].histogram
            if h.empty():
                continue
            hists, labels = splitHistogram(
                h,
                self.axis_options or None,
                return_labels=True,
            )
            style = None
            if self.style_set:
                style = self.style_set.getStyle(sector.sector_params)

            if isinstance(hists, dict):
                for c, h in hists.items():
                    everything.append(
                        PackagedHist(
                            histogram=h,
                            title=self.__getHistTitle(h, sector, dict(zip(labels, c))),
                            sector_parameters=sector.sector_params,
                            axis_options=self.axis_options,
                            style=style,
                        )
                    )
            else:
                everything.append(
                    PackagedHist(
                        histogram=hists,
                        title=self.__getHistTitle(hists, sector),
                        sector_parameters=sector.sector_params,
                        axis_options=self.axis_options,
                        style=style,
                    )
                )

        if self.add_together:
            ret = [
                PackagedHist(
                    histogram=ft.reduce(op.add, (x.histogram for x in everything)),
                    title="+".join(x.title for x in everything),
                    sector_parameters=sector.sector_params,
                    axis_options=self.axis_options,
                    style=everything[0].style,
                )
            ]
            return ret
        else:
            return everything

    def __rich_repr__(self):
        yield "parameters", self.parameters
        yield "axis_options", self.axis_options


def createSectorGroups(sectors, spec):
    if spec.to_process is not None:
        sectors = [x for x in sectors if spec.to_process.passes(x.sector_params)]
    grouped = groupBy(sectors, spec.fields, data_acquire=lambda x: x.params_dict)
    ret = [
        SectorGroup(
            parameters=params,
            sectors=sectors,
            axis_options=spec.axis_options,
            title_format=spec.title_format,
            cat_remap=spec.cat_remap,
            style_set=spec.style_set,
            add_together=spec.add_together,
        )
        for params, sectors in grouped
    ]
    return ret
