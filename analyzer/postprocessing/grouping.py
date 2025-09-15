from __future__ import annotations
import copy
import hist
from rich import print
from collections import defaultdict, OrderedDict, ChainMap
import itertools as it
import string
from typing import Annotated, Any, ClassVar
from analyzer.utils.querying import (
    NestedPatternExpression,
    modelIter,
    Pattern,
    PatternExpression,
    SimpleNestedPatternExpression,
)

from analyzer.core.results import SectorResult
from analyzer.core.specifiers import SectorParams
from pydantic import BaseModel, Field

from .split_histogram import Mode
from .style import Style
import logging


logger = logging.getLogger(__name__)


def doFormatting(s, **kwargs):
    parsed = string.Formatter().parse(s)
    s = ""
    for x in parsed:
        s += x[0]
        if x[1] is not None:
            s += str(kwargs[x[1]])
            # else:
            #     s += str(getNested(all_data, x[1]))
    return s


class HistogramProvenance(BaseModel):
    name: str

    sector_parameters: SectorParams = Field(repr=False)
    group_params: dict[str, Any]
    axis_params: dict[str, Any] = Field(default_factory=dict)
    merged_from: list[HistogramProvenance] | None = None

    def allEntries(self):
        return ChainMap(
            {"histogram_name": self.name},
            self.group_params,
            self.axis_params,
            dict(modelIter(self.sector_parameters)),
        )


class MergedHistogramProvenance(BaseModel):
    merged_from: list[HistogramProvenance | MergedHistogramProvenance] | None = None

    def allEntries(self):
        return ChainMap(*(x.allEntries() for x in self.merged_from))

    @property
    def sector_parameters(self):
        return self.merged_from[0].sector_parameters


class PackagedHist(BaseModel):
    histogram: Any
    title: str
    provenance: HistogramProvenance | MergedHistogramProvenance
    style: Style | None = None

    @property
    def sector_parameters(self):
        return self.provenance.sector_parameters

    @property
    def dim(self):
        return len(self.histogram.axes)

    def compatible(self, other):
        return self.histogram.axes == other.histogram.axes


class SectorGroupParameters(BaseModel):
    parameters: dict[Any, Any]
    axis_options: dict[str, Mode | str | int] | None = None

    @property
    def all_parameters(self):
        return {**self.parameters}

    def compatible(self, other):
        return self.parameters == other.parameters


def groupsMatch(group1, group2, fields):
    return all(group1.all_parameters[f] == group2.all_parameters[f] for f in fields)


class ScaleHistograms(BaseModel):
    limit: NestedPatternExpression

    def __call__(self, histograms):
        return histograms


class RemapCategories(BaseModel):
    nothing_here: None = None

    def __call__(self, histograms):
        return histograms


def _mergeHists(hists):
    ret = PackagedHist(
        histogram=sum(h.histogram for h in hists),
        provenance=MergedHistogramProvenance(merged_from=[x.provenance for x in hists]),
        style=hists[0].style,
        title=hists[0].title,
    )
    return ret


class Merge(BaseModel):
    merge_fields: list[str] | None = None

    def __call__(self, histograms):
        groups = defaultdict(list)
        if self.merge_fields is not None:
            for ph in histograms:
                captured = self.merge_fields.capture(ph.provenance.sector_parameters)
                groups[dictToFrozen(captured)].append(ph)
            return [_mergeHists(g) for g in groups.values()]
        else:
            return [_mergeHists(histograms)]
        


class SplitAxes(BaseModel):
    split_axis_names: list[str]
    pattern: dict[str, PatternExpression] | PatternExpression | None = None

    def __call__(self, histograms):
        ret = []
        for ph in histograms:
            h = ph.histogram

            split_axes = [h.axes[a] for a in self.split_axis_names]

            def passedPattern(name, val):
                if self.pattern is not None:
                    if isinstance(self.pattern, dict):
                        return self.pattern[name].match(val)
                    else:
                        return self.pattern.match(val)
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
                provenance = copy.deepcopy(ph.provenance)
                provenance.axis_params.update(axis_values)
                ret.append(
                    PackagedHist(
                        histogram=split_hist,
                        provenance=provenance,
                        style=ph.style,
                        title=ph.title,
                    )
                )

        return ret


class MergeAxes(BaseModel):
    merge_axis_names: list[str | int]

    def __call__(self, histograms):
        ret = []
        for ph in histograms:
            h = ph.histogram
            merging = {x: sum for x in self.merge_axis_names}
            h = h[merging]
            provenance = copy.deepcopy(ph.provenance)
            provenance.axis_params.update(merging)
            ret.append(
                PackagedHist(
                    histogram=h,
                    provenance=provenance,
                    style=ph.style,
                    title=ph.title,
                )
            )

        return ret


class SelectAxesValues(BaseModel):
    select_axes_values: dict[str, str | int | float]

    def __call__(self, histograms):
        ret = []
        for ph in histograms:
            h = ph.histogram
            h = h[self.select_axes_values]
            provenance = copy.deepcopy(ph.provenance)
            provenance.axis_params.update(self.select_axes_values)
            ret.append(
                PackagedHist(
                    histogram=h,
                    provenance=provenance,
                    style=ph.style,
                    title=ph.title,
                )
            )

        return ret


class RebinAxes(BaseModel):
    rebin: int | dict[str, int]

    def __call__(self, histograms):
        ret = []
        for ph in histograms:
            h = ph.histogram
            if isinstance(self.rebin, dict):
                rebins = {x: hist.rebin(y) for x, y in self.rebin}
            else:
                rebins = {x.name: hist.rebin(self.rebin) for x in h.axes}
            h = h[rebins]
            provenance = copy.deepcopy(ph.provenance)
            provenance.axis_params.update(rebins)
            ret.append(
                PackagedHist(
                    histogram=h,
                    provenance=provenance,
                    style=ph.style,
                    title=ph.title,
                )
            )

        return ret


class SliceAxes(BaseModel):
    slices: list[tuple[int | float | None, int | float | None]]

    def __call__(self, histograms):
        ret = []
        for ph in histograms:
            h = ph.histogram
            slices = dict(
                (a.name, slice(*(hist.loc(x) if x else x for x in s)))
                for a, s in zip(h.axes, self.slices)
            )
            h = h[slices]
            provenance = copy.deepcopy(ph.provenance)
            provenance.axis_params.update(slices)
            ret.append(
                PackagedHist(
                    histogram=h,
                    provenance=provenance,
                    style=ph.style,
                    title=ph.title,
                )
            )

        return ret


class FormatTitle(BaseModel):
    title_format: str

    def __call__(self, histograms):
        ret = []
        for ph in histograms:
            ret.append(
                PackagedHist(
                    histogram=ph.histogram,
                    provenance=ph.provenance,
                    style=ph.style,
                    title=doFormatting(self.title_format, **ph.provenance.allEntries()),
                )
            )
        return ret


AnyPipeline = (
    ScaleHistograms
    | RemapCategories
    | Merge
    | SplitAxes
    | RebinAxes
    | MergeAxes
    | SelectAxesValues
    | FormatTitle
    | SliceAxes
)


HistPipeline = list[AnyPipeline]


def dictToFrozen(d):
    return frozenset(sorted(d.items()))


class SectorPipelineSpec(BaseModel):
    group_fields: PatternExpression
    pipeline: HistPipeline

    def makePipelines(self, sectors):
        groups = defaultdict(list)
        for s in sectors:
            if self.group_fields.match(s.sector_params):
                captured = self.group_fields.capture(s.sector_params)
                groups[dictToFrozen(captured)].append(s)

        return [
            SectorHistPipeline(
                sector_group=SectorGroup(field_values=dict(k), sectors=s),
                pipeline=self.pipeline,
            )
            for k, s in groups.items()
        ]


class SectorHistPipeline(BaseModel):
    sector_group: SectorGroup
    pipeline: HistPipeline

    def getHists(self, name):
        ret = []
        for sector in self.sector_group.sectors:
            try:
                h = sector.result.histograms[name].histogram
            except KeyError as e:
                logger.error(
                    f"Could not find histogram '{name}' in {sector.sector_params.dataset.name} -- {sector.sector_params.region_name}"
                )
                raise e
            h = sector.result.histograms[name].histogram
            ph = PackagedHist(
                histogram=h,
                title="",
                provenance=HistogramProvenance(
                    name=name,
                    sector_parameters=sector.sector_params,
                    group_params=self.sector_group.field_values,
                ),
            )
            ret.append(ph)

        for p in self.pipeline:
            ret = p(ret)
        return ret


class SectorGroup(BaseModel):
    """
    A collection of sectors (Region,Datasets pairs), which are treated as a unit for certain processors purposes.
    Different processors may use this construction differently

    """

    separator: ClassVar[str] = " "

    field_values: dict[str, Any]
    sectors: Annotated[list[SectorResult], Field(repr=False)]

    def __iter__(self):
        return iter(self.sectors)

    # def __getHistTitle(self, hist, sector, cat_values=None):
    #     cat_values = cat_values or {}
    #     l = copy.deepcopy(cat_values)
    #     if self.cat_remap:
    #         for k, v in self.cat_remap.items():
    #             if k in l:
    #                 l[k] = v
    #     return doFormatting(
    #         self.title_format,
    #         **sector.sector_params.model_dump(),
    #         title=sector.sector_params.dataset.title,
    #     )

    # def getSectorStyle(self, sector):
    #     if self.style_set:
    #         return self.style_set.getStyle(sector.sector_params)
    #     else:
    #         return None
    #
    # def histograms(self, hist_name):
    #     """
    #     Get all the histograms corresponding to the given name.
    #     Returns a list of PackagedHist
    #     """
    #     everything = defaultdict(list)
    #     for sector in self.sectors:
    #         try:
    #             h = sector.result.histograms[hist_name].histogram
    #         except KeyError:
    #             logger.error(
    #                 f"Could not find histogram '{hist_name}' in {sector.sector_params.dataset.name} -- {sector.sector_params.region_name}"
    #             )
    #             raise
    #
    #         if h.empty():
    #             continue
    #         if self.rescale is not None:
    #             for s in self.rescale:
    #                 if s.sector_spec.passes(sector.sector_params):
    #                     logger.warn(
    #                         f"Scaling sector {sector.sector_params.simpleName()} by {s.scale} "
    #                     )
    #                     h = h * s.scale
    #                     break
    #
    #         hists, labels = splitHistogram(
    #             h,
    #             self.axis_options or None,
    #             return_labels=True,
    #         )
    #         style = None
    #         if self.style_set:
    #             style = self.style_set.getStyle(sector.sector_params)
    #
    #         if isinstance(hists, dict):
    #             for c, h in hists.items():
    #                 options = dict(zip(labels, c))
    #                 everything[c].append(
    #                     PackagedHist(
    #                         histogram=h,
    #                         title=self.__getHistTitle(h, sector, options),
    #                         sector_parameters=sector.sector_params,
    #                         axis_parameters=options,
    #                         style=style,
    #                     )
    #                 )
    #         else:
    #             everything[None].append(
    #                 PackagedHist(
    #                     histogram=hists,
    #                     title=self.__getHistTitle(hists, sector),
    #                     sector_parameters=sector.sector_params,
    #                     axis_parameters=self.axis_options,
    #                     style=style,
    #                 )
    #             )
    #
    #     if not self.add_together:
    #         return list(it.chain.from_iterable(everything.values()))
    #     else:
    #         ret = []
    #         for group in everything.values():
    #             new_name = "_plus_".join(
    #                 x.sector_parameters.dataset.name for x in group
    #             )
    #
    #             if self.add_titles:
    #                 new_title = "+".join(
    #                     x.sector_parameters.dataset.title for x in group
    #                 )
    #
    #             else:
    #                 new_title = group[0].sector_parameters.dataset.title
    #
    #             s = copy.deepcopy(sector.sector_params)
    #             s.dataset.name = new_name
    #             s.dataset.title = new_title
    #             ret.append(
    #                 PackagedHist(
    #                     histogram=ft.reduce(op.add, (x.histogram for x in group)),
    #                     title=new_title,
    #                     sector_parameters=s,
    #                     axis_parameters=group[0].axis_parameters,
    #                     style=group[0].style,
    #                 )
    #             )
    #         return ret
    #
    # def __rich_repr__(self):
    #     yield "parameters", self.parameters
    #     yield "axis_options", self.axis_options


def createSectorGroups(sectors, spec):
    if spec.to_process is not None:
        sectors = [x for x in sectors if spec.to_process.passes(x.sector_params)]
    grouped = groupBy(
        sectors,
        spec.fields,
        data_acquire=lambda x: x.params_dict,
    )
    grouped = combine(grouped, spec.fields, spec.special_add)
    ret = []

    for params, (special_add, sectors) in grouped:
        if special_add:
            s = [
                (x.sector_params.dataset.name, x.sector_params.region_name)
                for x in sectors
            ]
            print(f"Specially combining the following sectors: {s}")
        ret.append(
            SectorGroup(
                parameters=params,
                sectors=sectors,
                axis_options=spec.axis_options,
                title_format=spec.title_format,
                cat_remap=spec.cat_remap,
                style_set=spec.style_set,
                add_together=spec.add_together or special_add,
                add_titles=not special_add,
                rescale=spec.rescale,
            )
        )

    return ret


def joinOnFields(fields, *args, key=lambda x: x):
    pattern = SimpleNestedPatternExpression({f: Pattern.Any() for f in fields})
    matched = [[(dictToFrozen(pattern.capture(key(x))), x) for x in a] for a in args]
    ret = defaultdict(list)
    for k, v in matched[0]:
        if ret[k]:
            ret[k][0].append(v)
        else:
            ret[k] = [[v]]

    for m in matched[1:]:
        to_append = defaultdict(list)
        for capture, item in m:
            to_append[capture].append(item)
        for capture in ret:
            ret[capture].append(to_append[capture])
    # print(ret)
    return list(ret.values())
