# from __future__ import annotations
from dataclasses import dataclass, field, asdict
import inspect

import dataclasses as dcls
from typing import Optional, Any, get_origin, get_args, Union, Tuple, ClassVar
import itertools as it
from collections import namedtuple
import copy
from rich import print
import hist
import hist.dask as dah

import yaml

Hist = Union[hist.Hist, dah.Hist]

AnalysisSector = namedtuple("AnalysisSector", "sample_name region_name")

def isOptional(t):
    return get_origin(t) is Union and type(None) in get_args(t)

def getBase(t):
    if isOptional(t):
        base = get_args(t)[0]
    else:
        base = t
    return base

class DictifyMixin(object):
    shared_level = None
    key_is_field = None

    @classmethod
    def getFields(cls):
        return list(
            it.chain(
                dcls.fields(cls),
                *[
                    getBase(f.type).getFields()
                    for f in dcls.fields(cls)
                    if f.name in (cls.shared_level or [])
                ],
            )
        )

    @classmethod
    def tryFromDict(cls, t, data, key=None):
        if issubclass(t, DictifyMixin):
            if t.key_is_field is not None and key is not None:
                c = copy.deepcopy(data)
                c[t.key_is_field] = key
            else:
                c = data
            r = t.fromDict(c)
            return r
        else:
            return data

    @classmethod
    def decode(cls, t, data):
        top = get_origin(t)
        if top is dict:
            key_type, value_type = get_args(t)
            ret = {k: cls.tryFromDict(value_type, v, k) for k, v in data.items()}
        elif top is list:
            value_type = get_args(t)[0]
            ret = [cls.tryFromDict(value_type, v) for v in data]
        else:
            ret = data

        return ret

    @staticmethod
    def doSameLevel(field, data):
        t = field.type
        base = getBase(t)
        is_optional = isOptional(t)
        base_fields = base.getFields()
        contains = any(x.name in data for x in base_fields)
        if not contains and is_optional:
            return None
        else:
            return base.fromDict(data)

    @classmethod
    def fromDict(cls, data):
        print(f"Tryin fromDict on class {cls} with data:\n {data}")
        fields = dcls.fields(cls)

        shared = cls.shared_level or []

        args = {}

        def get(field, d):
            o = get_origin(field.type)
            found = d.get(field.name)
            if isOptional(field.type):
                return found
            else:
                if o is list:
                    return found or []
                elif o is dict:
                    return found or {}
                else:
                    return d[field.name]

        for field in fields:
            b = getBase(field.type)
            if inspect.isclass(b) and issubclass(b, DictifyMixin):
                if field.name in shared:
                    args[field.name] = DictifyMixin.doSameLevel(field, data)
                else:
                    args[field.name] = b.fromDict(data)
            else:
                gotten = get(field, data)
                if gotten is not None:
                    res = cls.decode(field.type, gotten)
                    args[field.name] = res

        return cls(**args)

    def toDict(self):
        ret = {}
        for k, v in asdict(self).items():
            if v is None:
                continue
            if k in (self.shared_level or []):
                ret.update(v.toDict())
            else:
                ret[k] = v
        return ret

@dataclass
class Histogram:
    spec: "HistogramSpec"
    histogram: Hist
    variations: dict[tuple[str, str], Hist]


@dataclass
class HistogramAxis(DictifyMixin):
    title: str
    type: str = 
    unit: Optional[str] = None
    description: Optional[str] = None

    

@dataclass
class HistogramSpec(DictifyMixin):
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
        for 


@dataclass
class SampleSpec(DictifyMixin):
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


@dataclass
class SectorSpec(DictifyMixin):
    shared_level: ClassVar[list[str]] = ["sample_spec"]

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


@dataclass
class ModuleCollection(DictifyMixin):
    spec: SampleSpec
    modules: list[str]


@dataclass
class Region(DictifyMixin):
    key_is_field: ClassVar[str] = "name"
    name: str
    selection: list[str] = field(default_factory=list)
    preselection: list[str] = field(default_factory=list)
    preselection_histograms: list[str] = field(default_factory=list)
    histograms: list[str] = field(default_factory=list)


@dataclass
class Weight(DictifyMixin):
    key_is_field: ClassVar[str] = "name"
    shared_level: ClassVar[list[str]] = ["sector_spec"]

    name: str
    sector_spec: SectorSpec
    variation: list[str] = field(default_factory=list)


@dataclass
class RegionDescription(DictifyMixin):
    name: str
    selelection: list[ModuleCollection] = field(default_factory=list)
    preselection: list[ModuleCollection] = field(default_factory=list)
    preselection_histograms: list[ModuleCollection] = field(default_factory=list)
    histograms: list[ModuleCollection] = field(default_factory=list)

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


@dataclass
class AnalysisDescription(DictifyMixin):
    name: str
    samples: dict[str, list[str]]
    regions: dict[str, Region]
    weights: dict[str, Weight]

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


pre_selection_histograms = {}
post_selection_histograms = {}
weights = {}
selectors = {}


def main():
    d = yaml.safe_load(open("test.yaml", "r"))
    an = AnalysisDescription.fromDict(d)
    print(an)


if __name__ == "__main__":
