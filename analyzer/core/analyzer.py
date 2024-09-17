# from __future__ import annotations
import copy
import enum
import functools as ft
import inspect
import itertools as it
import json
import logging
from collections import defaultdict, namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    ClassVar,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import yaml

import hist
import hist.dask as dah
from coffea.analysis_tools import PackedSelection
from rich import print

Hist = Union[hist.Hist, dah.Hist]


logger = logging.getLogger(__name__)


class AnalysisConfigurationError(Exception):
    def __init__(self, message):
        super().__init__(message)


@dataclass
class HistogramCollection:
    """A histogram needs to keep track of both its nominal value and any variations."""

    spec: "HistogramSpec"
    histogram: Hist
    variations: defaultdict[dict[str, Hist]]

    def __add__(self, other):
        if self.spec != other.spec:
            raise ValueError(f"Cannot add two incomatible histograms")
        return Histogram(
            self.spec,
            self.histogram + other.histogram,
            accumulate([self.variations, other.variations]),
        )


def generateHistogram(
    self,
    spec,
    fill_data,
    categories,
    weight_repo,
    weights=None,
    no_scale=False,
    mask=None,
):
    assert len(fill_data) == len(axes)
    base_histogram = dah.Hist(*self.axes, storage=self.storage, name=self.name)
    base_histogram.fill()
    variations = {}


@dataclass
class Selection:
    or_names: tuple[str] = field(default_factory=tuple)
    and_names: tuple[str] = field(default_factory=tuple)

    def addOne(self, name, type="and"):
        if type == "and":
            s = Selection(tuple(), (name,))
        else:
            s = Selection((name,), tuple())
        return self.merge(s)

    def merge(self, other):
        def addTuples(t, o):
            to_add = [x for x in o if x not in t]
            return tuple([*t, *to_add])

        return Selection(
            addTuples(self.or_names, other.or_names),
            addTuples(self.and_names, other.and_names),
        )


@dataclass
class Cutflow:
    n_minus_one: dict[str, float]
    cutflow: dict[str, float]


@dataclas
class SampleCutflow:
    unweighted_cutflow: Cutflow
    weighted_cutflow: Optional[Cutflow] = None


@dataclass
class SampleSelection:
    """
    Selection for a single sample.
    Stores the preselection and selection masks.
    The selection mask is relative to the "or" of all the preselection cuts.

    The general flow for a single sample looks like


    AllEvents -> Generate Preselection Masks -> AllEvents[PassAnyPreselectionCut]
    -> Generate Selection Mask -> Add Preselection Masks to Selection Mask
    -> Apply Appropriate Cuts for Each Region -> Final Trimmed Events

    """

    preselection_mask: PackedSelection = field(default_factory=PackedSelection)
    selection_mask: PackedSelection = field(default_factory=PackedSelection)

    def getPreselectionMask(self):
        names = self.preselection_mask.names
        if not names:
            return None
        return self.preselection_mask.any(*names)

    def addPreselectionMaskToSelection(self, complete_mask):
        """After finalizing selection  we want to add the preselection cuts to the analysis selection, so that we can generate cutflows etc.
        This function takes each preselection mask and adds the appropriate portion to the final selection, based on complete_mask, which should be the mask used to the compute the events post-preselection (ie the events that are ultimately selected on)
        """

        for n in self.preselection_mask.names:
            if complete_mask is None:
                self.addMask(n, self.preselection_mask.any(n))
            else:
                self.addMask(n, self.preselection_mask.any(n)[complete_mask])

    def addMask(self, name, mask, type="and", stage=AnalysisStage.Preselection):
        if stage == AnalysisStage.Preselection:
            target = self.preselection_mask
        else:
            target = self.selection_mask

        names = target.names
        if not name in names:
            target.add(name, mask)

    def getMask(self, or_name, and_names, stage=AnalysisStage.Preselection):
        if not (or_names or and_names):
            return None
        if stage == AnalysisStage.Preselection:
            sel = self.preselection_mask
        else:
            sel = self.selection_mask

        names = sel.names
        packed = self.preselections[k]
        t = packed.any(*or_names) if or_names else None
        a = packed.all(*ane_names) if and_names else None
        return ft.reduce(op.and_, (x for x in [t, a] if x is not None))


@dataclass
class SelectionManager:
    selection_masks: defaultdict[str, SampleSelection] = field(
        default_factory=lambda: defaultdict(SampleSelection)
    )

    selections: defaultdict[SectorId, Selection] = field(
        default_factory=lambda: defaultdict(Selection)
    )

    __computed_preselections: dict[str, Any] = field(default_factor=dict)

    def register(
        self, sector_id, name, mask, type="and", stage=AnalysisStage.Preselection
    ):
        self.selections[sector_id] = self.selections[sector_id].addOne(name, type=type)
        self.selections[sample_name].addMask(name, mask, stage=stage)

    def maskPreselection(self, sample_name, events):
        mask = self.selections[sample_name].getPreselectionMask(name)
        self.__computed_preselections[sample_name] = mask
        if mask is None:
            return events
        else:
            return events[mask]

    def addPreselectionMasks(self):
        for sample_name, sel in selection_mask.items():
            m = self.__computed_preselections[sample_name]
            sel.addPreselectionMaskToSelection(m)

    def maskSector(self, sector_id, events):
        s = self.selections[sector_id]
        mask = self.selections[sample_name].getMask(
            name, s.or_names, s.and_names, stage=AnalysisStage.Selection
        )
        if mask is None:
            return events
        else:
            return events[mask]


@dataclass
class WeightManager:
    weights: defaultdict[SectorId, Weight] = field(
        default_factory=lambda: defaultdict(Weight)
    )

    __cache: dict[(SectorId, str), Any] = field(default_factory=dict)

    def add(self, sector_id, weight_name, central, variations):
        systs = [(x, *y) for x, y in variations.items()]
        name, up, down = list(map(list, zip(*variations)))
        logger.debug(f"Weight {wname} has variations {', '.join(name)}")
        self.weights[sector_id].add_multivariation(weight_name, central, name, up, down)

    def weight(self, sector_id, modifier):
        k = (sector_id, modifier)
        if k in self.__cache:
            return self.__cache[k]
        else:
            weights = self.weights[sample_name].weight(modifier)
            self.__cache[k] = weights
            return weights


def getSectionHash(am_descs):
    to_hash = [{"name": x.name, "configuration": x.configuration} for x in am_descs]
    return hash(json.dumps(to_hash, sort_keys=True))


@dataclass
class Sector:
    region_name: str
    sample_name: str

    description: str = ""

    forbid_data: bool = False

    # Parts of the analysis.
    preselection: list[AnalyzerModule] = field(default_factory=list)
    objects: list[AnalyzerModule] = field(default_factory=list)
    selection: list[AnalyzerModule] = field(default_factory=list)
    categories: list[AnalyzerModule] = field(default_factory=list)
    # preselection_histograms: list[AnalyzerModule] = field(default_factory=list)
    histograms: list[AnalyzerModule] = field(default_factory=list)
    weights: list[AnalyzerModule] = field(default_factory=list)

    # Dictionary of weight -> [variations] to user for this sector
    # weights: dict[str, Optional[list]] = field(default_factory=dict)

    @property
    def sector_id(self):
        return SectorId(sample_name, region_name)

    def getSelectionId(self):
        return getSectionHash(self.selection)

    def getPreselectionId(self):
        return getSectionHash(self.selection)

    @staticmethod
    def fromRegion(region_desc, sample, module_repo, weight_repo):
        name = region_desc.name

        if region_desc.forbid_data and sample.sample_type == "Data":
            raise AnalysisConfigurationError(
                f"Region '{region_desc.name}' is marked with 'forbid_data'"
                f"but is recieving Data sample '{sample.name}'"
            )

        def resolveModules(l, t):
            ret = [
                mod
                for mod in l
                if not mod.sample_spec or mod.sample_spec.passes(sample)
            ]
            modules = [
                [
                    module_repo.get(t, mod.name, configuration=c)
                    for c in (
                        mod.config if isinstance(mod.config, list) else [mod.config]
                    )
                ]
                for mod in ret
            ]
            ret = list(it.chain(*modules))
            return ret

        preselection = resolveModules(region_desc.preselection, ModuleType.Selection)
        selection = resolveModules(region_desc.selection, ModuleType.Selection)
        objects = resolveModules(region_desc.selection, ModuleType.Producer)
        preselection_histograms = resolveModules(
            region_desc.preselection_histograms, ModuleType.Histogram
        )
        postselection_histograms = resolveModules(
            region_desc.histograms, ModuleType.Histogram
        )
        weights = weight_repo.getWeightsForSector(sample.name, region.name)

        return Sector(
            region_name=region_desc.name,
            sample_name=sample.name,
            description=region_desc.description,
            forbid_data=region_desc.forbid_data,
            preselection=preselection,
            selection=selection,
            objects=objects,
            preselection_histograms=preselection_histograms,
            histograms=postselection_histograms,
            weights=weights,
        )


@dataclass(frozen=True)
class SectorId:
    sample_name: str
    region_name: str


class RegionDescription(BaseModel):
    name: str
    forbid_data: bool = False
    description: str = ""
    selection: list[ModuleDescription] = Field(default_factory=list)
    objects: list[ModuleDescription] = Field(default_factory=list)
    preselection: list[ModuleDescription] = Field(default_factory=list)
    preselection_histograms: list[ModuleDescription] = Field(default_factory=list)
    histograms: list[ModuleDescription] = Field(default_factory=list)


class AnalysisDescription(BaseModel):
    """Description of an analysis"""

    name: str

    description: str

    samples: dict[str, Union[list[str], str]]
    regions: list[RegionDescription]
    weights: list[Weight]
    general_config: dict[str, Any] = Field(default_factory=dict)

    # Names of regions to reserve
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
class Analyzer:
    description: AnalysisDescription

    datasets_preprocessed: dict[str, "DatasetPreprocessed"] = field(
        default_factory=dict
    )

    sample_events: dict[str, Any] = None
    preselected_events: dict[str, Any] = None
    sector_events: dict[SectorId, Any] = None

    weight_manager: WeightManager = None
    selection_manager: SelectionManager = None

    sample_repo: "SampleRepo" = None
    era_repo: "EraRepo" = None

    def getSectorsAnalyzers(self):
        sector_pairs = self.description.getSectors()
        for sample_name, region_name in sector_pairs:
            sample = sample_repo.get(sample_name)
            region = self.description.getRegion(region_name)

    def getEvents(sample_name, stage=None):
        return self.sample_events[sample_name]


@dataclass
class Category:
    name: str
    axis: Any
    values: Any


@dataclass
class AnalysisResult:
    datasets_preprocessed: dict[str, "DatasetPreprocessed"]
    processed_chunks: dict[str, set[Chunk]]
    description: AnalysisDescription
    results: dict[tuple[str, str], SectorResult]


@dataclass
class SectorResult:
    histograms: dict[str, HistogramCollection]
    other_data: dict[str, Any]
    cutflow_data: Optional[Any]


@dataclass
class SectorAnalyzer:
    analyzer: Analyzer
    sample: "Sample"
    sector: Sector
    result: SectorResult
    sector_selection: Selection
    selection_stage: str

    categories: dict[str, Category] = field(default_factory=dict)

    @property
    def sector_id(self):
        return region.sector_id

    @property
    def events(self):
        return self.analyzer.sample_events[self.sample.name]

    def makeHistogram(self, spec: HistogramSpec, values):
        """Add a histogram based on a specification and the values to fill it."""
        pass

    def addHistogram(self, hist):
        """Add an existing histogram"""
        pass

    def addCategory(self, category):
        self.categories[category.name] = category

    def addSelection(self, name, mask, type="and", stage="preselection"):
        self.analyzer.selection_manager.register(
            self.sector_id, name, mask, type=type, stage=stage
        )

    def addWeight(self, name, central, variations=None):
        varia = variations or {}
        self.analyzer.weight_manager.add(self.sector_id, name, centra, varia)

    def getParams(self):
        pass

    class Selector:
        def __init__(self, parent, stage):
            self.parent = parent
            self.stage = stage

        def add(self, name, mask, type="and"):
            return self.parent.addSelection(name, mask, type=type, stage=self.stage)

    class Weighter:
        def __init__(self, parent):
            self.parent = parent

        def add(self, *args, **kwargs):
            return self.parent.addWeight(*args, **kwargs)

    class Categorizer:
        def __init__(self, parent):
            self.parent = parent

        def add(self, *args, **kwargs):
            return self.parent.addCategory(*args, **kwargs)

    class Histogrammer:
        def __init__(self, parent):
            self.parent = parent

        def addHistogram(self, *args, **kwargs):
            return self.parent.addHistogram(*args, **kwargs)

        def makeHistogram(self, *args, **kwargs):
            return self.parent.makeHistogram(*args, **kwargs)

    def __getStageProcessor(self, stage):
        mapping = {
            AnalysisStage.Preselection: Selector(self, "preselection"),
            AnalysisStage.Selection: Selector(self, "selection"),
            AnalysisStage.Categorization: Categorizer(self),
            AnalysisStage.Weight: Weighter(self),
            AnalysisStage.Histogram: Histogrammer(self),
        }
        return mapping.get(stage)

    def applyToEvents(self, events, stage):
        sp = self.__getStageProcessor(stage)
        mapping = {
            AnalysisStage.Preselection: self.region.preselection,
            AnalysisStage.Selection: self.region.selection,
            AnalysisStage.Categorization: self.region.cateogories,
            AnalysisStage.Histogram: self.region.histograms,
            AnalysisStage.Objects: self.region.objects,
            AnalysisStage.Weights: self.region.weights,
        }
        params = {}

        for module in mapping[stage]:
            if sp:
                module(events, params, sp)
            else:
                module(events, params)


def runAnalysis(analyzer):

    sector_analyzers = analyzer.getSectorAnalyzers()

    for d in analyzer.datasets_preprocessed:
        analyzer.sample_events[d.sample_name] = getEvents(d)

    for sa in sector_analyzers:
        events = analyzer.getEvents(sa.sector.sample_name)
        sa.applyToEvents(events, AnalysisStage.Preselection)

    for sample in self.sample_events:
        events = analyzer.getEvents(sa.sector.sample_name)
        analyzer.preselected_events[
            sample
        ] = analyzer.selection_manager.maskPreselection(events)

    for sa in sector_analyzers:
        events = analyzer.preselected_events[sa.sector.sample_name]
        sa.applyToEvents(events, AnalysisStage.Objects)

    for sa in sector_analyzers:
        events = analyzer.preselected_events[sa.sector.sample_name]
        sa.applyToEvents(events, AnalysisStage.Selection)

    for sa in sector_analyzer:
        events = analyzer.preselected_events[sa.sector.sample_name]
        analyzer.selected_events[sa.sector_id] = analyzer.selection_manager.maskSector(
            sector_id, events
        )

    for sa in sector_analyzers:
        events = analyzer.selected_events[sa.sector_id]
        sa.applyToEvents(events, AnalysisStage.Weights)

    for sa in sector_analyzers:
        events = analyzer.selected_events[sa.sector_id]
        sa.applyToEvents(events, AnalysisStage.Categorization)

    for sa in sector_analyzers:
        events = analyzer.selected_events[sa.sector_id]
        sa.applyToEvents(events, AnalysisStage.Histograms)


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
