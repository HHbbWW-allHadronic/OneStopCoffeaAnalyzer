# from __future__ import annotations
import inspect
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass, field
from coffea.analysis_tools import PackedSelection


from typing import (
    Optional,
    Any,
    get_origin,
    get_args,
    Union,
    Tuple,
    ClassVar,
    Annotated,
)
import itertools as it
from collections import namedtuple
import copy
from rich import print
import hist
import hist.dask as dah

import yaml

Hist = Union[hist.Hist, dah.Hist]

AnalysisSector = namedtuple("AnalysisSector", "sample_name region_name")


from typing import Annotated, Any

from pydantic import BaseModel, GetCoreSchemaHandler, TypeAdapter, create_model
from pydantic_core import CoreSchema, core_schema


@dataclass
class Histogram:
    spec: "HistogramSpec"
    histogram: Hist
    variations: dict[tuple[str, str], Hist]


@dataclass
class HistogramAxis:
    title: str
    type: str
    unit: Optional[str] = None
    description: Optional[str] = None


class HistogramSpec(BaseModel):
    name: str
    axes: list[HistogramAxis]
    storage: str = "weight"
    description: str
    weights: list[str] = None
    weight_variations: dict[str, list[str]]

    def generateHistogram(sector, fill_data, weight_repo, mask=None):
        assert len(fill_data) == len(axes)
        base_histogram = dah.Hist(*self.axes, storage=self.storage, name=self.name)
        base_histogram.fill()
        variations = {}
        weights = self.weights or []


class SampleSpec(BaseModel):
    sample_names: Optional[Union[list[str], str]] = None
    eras: Optional[Union[list[str], str]] = None
    sample_types: Optional[Union[list[str], str]] = None

    def passes(sample, sample_manager):
        sample = sample_manager.get(sample)
        passes = True
        passes_names = not self.sample_names or any(
            sample.name == x for x in self.sample_names
        )
        passes_era = not self.sample_named or any(sample.era == x for x in self.eras)
        passes_type = not self.sample_types or any(
            sample.sample_type == x for x in self.sample_types
        )
        return passes_names and passes_era and passes_type


class SectorSpec(BaseModel):
    sample_spec: Optional[SampleSpec] = None
    region_names: Optional[Union[list[str], str]] = None

    def passes(sector, sample_manager):
        passes_sample = not self.sample_spec or sample_spec.passes(
            sector.sample_name, sample_manager
        )
        passes_region = not self.region_names or any(
            sector.region_name == x for x in self.region_names
        )
        return passes_sample and passes_region


class ModuleDescription(BaseModel):
    name: str
    sample_spec: Optional[SampleSpec] = None
    config: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None


class Region(BaseModel):
    name: str
    forbid_data: bool = False
    selection: list[str] = Field(default_factory=list)
    preselection: list[str] = Field(default_factory=list)
    preselection_histograms: list[str] = Field(default_factory=list)
    histograms: list[str] = Field(default_factory=list)


class Weight(BaseModel):
    name: str
    sector_spec: Optional[SectorSpec] = None
    variation: list[str] = Field(default_factory=list)
    region_config: Optional[dict[str, dict[str, Any]]] = None


@dataclass
class Selection:
    triggers: set[str] = field(default_factory=set)
    analysis_cuts: set[str] = field(default_factory=set)

    def merge(self, other):
        return Selection(
            self.triggers | other.triggers, self.analysis_cuts | other.analysis_cuts
        )


@dataclass
class SelectionManager:
    preselections: dict[str, PackedSelection]
    preselection_regions: dict[str, Selection]
    selections: dict[(str, int), PackedSelection]
    selection_regions: dict[str, Selection]

    def registerOneCut(
        self, sample_name, name, mask, sel, reg, type, preselection_hash=None
    ):
        k = (
            sample_name
            if preselection_hash is None
            else (sample_name, preselection_hash)
        )
        packed = sel.getdefault(k, PackedSelection())
        s = reg.getdefault(sample_namek, Selection())
        if name not in packed.names:
            packed.add(name, mask)
        if type == "trigger":
            s.triggers.add(name)
        else:
            s.analysis_cuts.add(name)

    def register(
        self, sample_name, name, mask, preselection_hash=None, type="analysis_cut"
    ):
        if preselection_hash is None:
            self.registerOneCut(
                sample_name,
                name,
                mask,
                self.preselections,
                self.preselection_regions,
                type=type,
            )
        else:
            self.registerOneCut(
                sample_name,
                name,
                mask,
                self.selections,
                self.selection_regions,
                type=type,
            )

    def getMask(self, events, sample_name, region_name, preselection_hash=None):
        k = (
            sample_name
            if preselection_hash is None
            else (sample_name, preselection_hash)
        )
        sel = self.preselection_region.get(region_name)
        if sel is None:
            return ak.ones_like(events)
        triggers, analysis_cuts = sel.triggers, sel.analysis_cuts
        packed = self.preselections[k]
        t = packed.any(*triggers) if triggers else None
        a = packed.all(*analysis_cuts) if analysis_cuts else None
        return ft.reduce(op.and_, (x for x in [t, a] if x is not None))


class RegionDescription(BaseModel):
    name: str
    selelection: list[ModuleDescription] = Field(default_factory=list)
    preselection: list[ModuleDescription] = Field(default_factory=list)
    preselection_histograms: list[ModuleDescription] = Field(default_factory=list)
    histograms: list[ModuleDescription] = Field(default_factory=list)

    def getRegionForSample(self, sample, sample_manager):
        name = self.name

        def doFilter(l):
            ret = list(
                it.chain.from_iterable(
                    [coll.modules for x in l if coll.spec.passes(sample)]
                )
            )
            return ret

        pre_selection = doFilter(self.pre_selection)
        selection = doFilter(self.selection)
        pre_selection_histograms = doFilter(self.pre_selection_selection)
        post_selection_histograms = doFilter(self.post_selection_selection)
        return Region(
            self.name,
            self.desc,
            pre_selection,
            selection,
            pre_selection_histograms,
            post_selection_histograms,
        )

    # __dsk: Optional[Any] = None

    # def __dask_graph__(self):
    #    if self.__dsk is not None:
    #        return self.__dsk
    #    else:
    #        r =  collections_to_dsk(tuple(self.spec, self.histogram, self.variations))
    #        self.__dsk = r
    #        return r

    # def __dask_keys__(
    #    if self.__dsk  not None:
    #        r =  collections_to_dsk(tuple(self.spec, self.histogram, self.variations))
    #        self.__dsk = r
    #    return  self.__dsk.keys()


class AnalysisDescription(BaseModel):
    name: str
    object_definitions: list[ModuleDescription]
    samples: dict[str, list[str]]
    regions: list[RegionDescription]
    weights: list[Weight]

    def getAnalysisSectors(self):
        ret = []
        for sample_name, regions in self.samples:
            for r in regions:
                ret.append(AnalysisSector(sample_name, region))
        return ret


Chunk = namedtuple("Chunk", "file start end")


@dataclass
class SectorResult:
    histograms: dict[str, Histogram]
    other_data: dict[str, Any]
    cutflow_data: Any


@dataclass
class AnalysisResult:
    datasets_preprocessed: dict[str, "DatasetPreprocessed"]
    processed_chunks: dict[str, set[Chunk]]

    description: AnalysisDescription
    results: dict[AnalysisSector, SectorResult]


def main():
    d = yaml.safe_load(open("pydtest.yaml", "r"))
    print(d)
    an = AnalysisDescription(**d)
    print(an)


if __name__ == "__main__":
    main()
