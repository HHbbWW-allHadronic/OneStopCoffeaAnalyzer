# from __future__ import annotations
import inspect
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass, field
from coffea.analysis_tools import PackedSelection
from collections.abc import Callable
import enum
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
import json
from collections import namedtuple, defaultdict
import copy
from rich import print
import hist
import hist.dask as dah
from typing import Annotated, Any
from pydantic import BaseModel, GetCoreSchemaHandler, TypeAdapter, create_model
from pydantic_core import CoreSchema, core_schema
import yaml

Hist = Union[hist.Hist, dah.Hist]


@dataclass
class Histogram:
    """A histogram needs to keep track of both its nominal value and any variations."""

    spec: "HistogramSpec"
    histogram: Hist
    variations: dict[tuple[str, str], Hist]


class HistogramSpec(BaseModel):
    name: str
    axes: list[Any]
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

    def passes(self, sample):
        passes_names = not self.sample_names or any(
            sample.name == x for x in self.sample_names
        )
        passes_era = not self.eras or any(sample.era == x for x in self.eras)
        passes_type = not self.sample_types or any(
            sample.sample_type == x for x in self.sample_types
        )
        return passes_names and passes_era and passes_type


class ModuleDescription(BaseModel):
    name: str
    sample_spec: Optional[SampleSpec] = None
    config: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None


class ModuleType(str, enum.Enum):
    Selection = "Selection"
    Categorization = "Categorization"
    Weight = "Weight"
    Histogram = "Histogram"
    Producer = "Producer"


@dataclass
class AnalyzerModuleDescription:
    name: str
    type: ModuleType
    configuration: dict[str, Any] = field(default_factory=dict)


class SectorSpec(BaseModel):
    sample_spec: Optional[SampleSpec] = None
    region_names: Optional[Union[list[str], str]] = None

    def passes(self, sector):
        passes_sample = not self.sample_spec or sample_spec.passes(
            sector.sample_name, sample_manager
        )
        passes_region = not self.region_names or any(
            sector.region_name == x for x in self.region_names
        )
        return passes_sample and passes_region


class ExpansionMode(str, enum.Enum):
    product = "product"
    zip = "zip"


class Weight(BaseModel):
    name: str
    sector_spec: Optional[SectorSpec] = None
    variation: list[str] = Field(default_factory=list)
    region_config: Optional[dict[str, dict[str, Any]]] = None


# @dataclass
# class Selection:
#     parent: "Selection"
#     triggers: set[str] = field(default_factory=set)
#     analysis_cuts: set[str] = field(default_factory=set)


#     def merge(self, other):
#         return Selection(
#             self.triggers | other.triggers, self.analysis_cuts | other.analysis_cuts
#         )


@dataclass
class Selection:
    masks: dict[str, Any] = field(default_factory=dict)
    parent: Optional["Selection"] = None

    @property
    def names(self):
        parent_names = parent.names if parent is not None else set()
        this_names = set(self.masks.names)
        return this_names | parent_names

    def add(self, name, mask):
        if name in self.names:
            raise ValueError(f"Name {name} is already in selection")
        self.masks[name] = mask

    def generatePackedSelection(self):
        p = PackedSelection()
        for n, m in masks.items():
            p.add(n, m)
        return p


@dataclass
class Selection:
    or_names: tuple[str] = field(default_factory=tuple)
    and_names: tuple[str] = field(default_factory=tuple)

    def merge(self, other):
        def addTuples(t, o):
            to_add = [x for x in o if x not in t]
            return tuple([*t, *to_add])

        return Selection(
            addTuples(self.or_names, other.or_names),
            addTuples(self.and_names, other.and_names),
        )


@dataclass
class SampleSelection:
    preselection_mask: PackedSelection = field(default_factory=PackedSelection)
    selection_mask: PackedSelection = field(default_factory=PackedSelection)

    def getPreselectionMask(self):
        return self.preselection_mask.any(*self.preselection_mask.names)

    def addPreselectionMaskToSelection(self, complete_mask):
        for n in self.preselection_mask.names:
            self.addMask(n, self.preselection_mask.any(n)[complete_mask])

    def addMask(self, presel_id, name, mask, type="and", stage="preselection"):
        if stage == "preselection":
            target = self.preselection_mask
        else:
            target = self.selection_mask

        names = target.names
        if not name in names:
            target.add(name, mask)

    def getMask(self, events, or_name, and_names, stage="preselection"):
        if stage == "preselection":
            sel = self.preselection_mask
        else:
            sel = self.selection_mask
        names = sel.names
        if not names:
            return ak.ones_like(events)
        packed = self.preselections[k]
        t = packed.any(*or_names) if or_names else None
        a = packed.all(*ane_names) if and_names else None
        return ft.reduce(op.and_, (x for x in [t, a] if x is not None))


@dataclass
class SelectionManager:
    selections: defaultdict[str, SampleSelection] = field(
        default_factory=lambda: defaultdict(SampleSelection)
    )

    def register(self, sample_name, name, mask, stage="preselection"):
        self.selections[sample_name].addMask(name, mask, stage=stage)

    def getMask(self, events, sample_name, or_names, and_names, stage="preselection"):
        self.selections[sample_name].getMask(
            events, name, or_names, and_names, stage=stage
        )


@dataclass
class AnalyzerModule:
    name: str
    function: Optional[Callable] = None
    configuration: dict[str, Any] = field(default_factory=dict)
    documentation: str = ""

    def __call__(self, events, analyzer):
        return self.function(events, analyzer, **self.configuration)


def getSectionHash(am_descs):
    to_hash = [{"name": x.name, "configuration": x.configuration} for x in am_descs]
    return hash(json.dumps(to_hash, sort_keys=True))


@dataclass
class SectorRegion:
    name: str
    description: str = ""
    forbid_data: bool = False
    selection: list[AnalyzerModule] = Field(default_factory=list)
    objects: list[AnalyzerModule] = Field(default_factory=list)
    preselection: list[AnalyzerModule] = Field(default_factory=list)
    preselection_histograms: list[AnalyzerModule] = Field(default_factory=list)
    histograms: list[AnalyzerModule] = Field(default_factory=list)

    def getSelectionId(self):
        return getSectionHash(self.selection)

    def getPreselectionId(self):
        return getSectionHash(self.selection)


class RegionDescription(BaseModel):
    name: str
    forbid_data: bool = False
    description: str = ""
    selection: list[ModuleDescription] = Field(default_factory=list)
    objects: list[ModuleDescription] = Field(default_factory=list)
    preselection: list[ModuleDescription] = Field(default_factory=list)
    preselection_histograms: list[ModuleDescription] = Field(default_factory=list)
    histograms: list[ModuleDescription] = Field(default_factory=list)

    def getSector(self, sample, module_repo):
        name = self.name

        def doFilter(l, t):
            ret = [
                mod
                for mod in l
                if not mod.sample_spec or mod.sample_spec.passes(sample)
            ]
            modules = [
                [
                    module_repo.create(t, mod.name, configuration=c)
                    for c in (
                        mod.config if isinstance(mod.config, list) else [mod.config]
                    )
                ]
                for mod in ret
            ]
            ret = list(it.chain(*modules))
            return ret

        preselection = doFilter(self.preselection, ModuleType.Selection)
        selection = doFilter(self.selection, ModuleType.Selection)
        objects = doFilter(self.selection, ModuleType.Producer)
        preselection_histograms = doFilter(
            self.preselection_histograms, ModuleType.Histogram
        )
        postselection_histograms = doFilter(self.histograms, ModuleType.Histogram)

        return SectorRegion(
            name=self.name,
            description=self.description,
            forbid_data=self.forbid_data,
            preselection=preselection,
            selection=selection,
            objects=objects,
            preselection_histograms=preselection_histograms,
            histograms=postselection_histograms,
        )


class AnalysisDescription(BaseModel):
    name: str
    object_definitions: list[ModuleDescription]
    samples: dict[str, Union[list[str], str]]
    regions: list[RegionDescription]
    weights: list[Weight]
    general_config: dict[str, Any] = Field(default_factory=dict)

    special_region_name: ClassVar[tuple[str]] = ("All",)

    def getSectors(self):
        ret = []
        for sample_name, regions in self.samples.items():
            if isinstance(regions, str) and regions == "All":
                regions = [r.name for r in self.regions]
            for r in regions:
                ret.append((sample_name, r))
        return ret

    def getRegion(self, name):
        try:
            return next(x for x in self.regions if x.name == name)
        except StopIteration as e:
            raise KeyError(f'No region "{name}"')

    def getWeight(self, name):
        try:
            return next(x for x in self.weights if x.name == name)
        except StopIteration as e:
            raise KeyError(f'No region "{name}"')


Chunk = namedtuple("Chunk", "file start end")


@dataclass
class SectorResult:
    histograms: dict[str, Histogram]
    other_data: dict[str, Any]
    cutflow_data: Any


@dataclass
class Analyzer:
    description: AnalysisDescription
    datasets_preprocessed: dict[str, "DatasetPreprocessed"] = field(
        default_factory=dict
    )
    sample_events: dict[str, Any] = None

    weight_manager: "WeightManager" = None
    selection_manager: SelectionManager = None

    sample_repo: "SampleManager" = None
    era_repo: "EraRepo" = None


@dataclass
class SectorAnalyzer:
    analyzer: Analyzer
    sample: "Sample"
    region: SectorRegion

    #    @property
    #    def region(self):
    #        return self.region
    #
    #    @property
    #    def sample(self):
    #        return self.sample

    @property
    def events(self):
        return self.analyzer.sample_events[self.sample.name]


@dataclass
class AnalysisResult:
    datasets_preprocessed: dict[str, "DatasetPreprocessed"]
    processed_chunks: dict[str, set[Chunk]]

    description: AnalysisDescription
    results: dict[tuple[str, str], SectorResult]


class FakeRepo:
    def create(self, type, name, **kwargs):
        return dict(type=type, name=name, **kwargs)


def runAnalysisDescription(desc):

    analyzer = Analyzer(desc)
    sas = []

    # for n,d in analyzer.datasets_preprocessed.items():
    #    analyzer.sample_events[n] = getEvents(d)

    for sample_name, region_name in desc.getSectors():
        region = desc.getRegion(region_name)
        sec = region.getSector(sample_name, FakeRepo())
        sas.append(SectorAnalyzer(analyzer, sample_name, sec))

    for s in sas:
        print("=========================")
        print(s.region)
        for p in s.region.preselection:
            print(f"Processing preselection {p}")

        for p in s.region.objects:
            print(f"Processing preselection {p}")


def main():
    d = yaml.safe_load(open("pydtest.yaml", "r"))
    an = AnalysisDescription(**d)
    runAnalysisDescription(an)


if __name__ == "__main__":
    main()
