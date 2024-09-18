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

import hist
import hist.dask as dah
import yaml
from analyzer.datasets import Dataset, Era, SampleId
from coffea.analysis_tools import PackedSelection, Weights
from rich import print
import operator as op

from .analysis_modules import MODULE_REPO, AnalyzerModule, ModuleType
from .configuration import AnalysisDescription, AnalysisStage, HistogramSpec
from .preprocessed import SamplePreprocessed, preprocessBulk

Hist = Union[hist.Hist, dah.Hist]


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


@dataclass(frozen=True)
class SectorId:
    sample_id: SampleId
    region_name: str


@dataclass
class Cutflow:
    n_minus_one: dict[str, float]
    cutflow: dict[str, float]


@dataclass
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
            print(n)
            if complete_mask is None:
                self.addMask(
                    n, self.preselection_mask.any(n), stage=AnalysisStage.Selection
                )
            else:
                self.addMask(
                    n,
                    self.preselection_mask.any(n)[complete_mask],
                    stage=AnalysisStage.Selection,
                )

    def addMask(self, name, mask, type="and", stage=AnalysisStage.Preselection):
        if stage == AnalysisStage.Preselection:
            target = self.preselection_mask
        else:
            target = self.selection_mask

        names = target.names
        if not name in names:
            target.add(name, mask)

    def getMask(self, or_names, and_names, stage=AnalysisStage.Preselection):
        if not (or_names or and_names):
            return None
        if stage == AnalysisStage.Preselection:
            sel = self.preselection_mask
        else:
            sel = self.selection_mask

        names = sel.names
        print(or_names)
        print(and_names)
        print(sel)
        t = sel.any(*or_names) if or_names else None
        a = sel.all(*and_names) if and_names else None
        return ft.reduce(op.and_, (x for x in [t, a] if x is not None))


@dataclass
class SelectionManager:
    selection_masks: defaultdict[SampleId, SampleSelection] = field(
        default_factory=lambda: defaultdict(SampleSelection)
    )

    selections: defaultdict[SectorId, Selection] = field(
        default_factory=lambda: defaultdict(Selection)
    )

    __computed_preselections: dict[str, Any] = field(default_factory=dict)

    def register(
        self, sector_id, name, mask, type="and", stage=AnalysisStage.Preselection
    ):
        self.selections[sector_id] = self.selections[sector_id].addOne(name, type=type)
        self.selection_masks[sector_id.sample_id].addMask(name, mask, stage=stage)

    def maskPreselection(self, sample_id, events):
        mask = self.selection_masks[sample_id].getPreselectionMask()
        self.__computed_preselections[sample_id] = mask
        if mask is None:
            return events
        else:
            return events[mask]

    def addPreselectionMasks(self):
        for sample_id, sel in self.selection_masks.items():
            m = self.__computed_preselections[sample_id]
            sel.addPreselectionMaskToSelection(m)

    def maskSector(self, sector_id, events):
        s = self.selections[sector_id]
        sample_id = sector_id.sample_id
        mask = self.selection_masks[sample_id].getMask(
            s.or_names, s.and_names, stage=AnalysisStage.Selection
        )
        if mask is None:
            return events
        else:
            return events[mask]


@dataclass
class WeightManager:
    weights: defaultdict[SectorId, Weights] = field(
        default_factory=lambda: defaultdict(lambda: Weights(None))
    )

    __cache: dict[(SectorId, str), Any] = field(default_factory=dict)

    def add(self, sector_id, weight_name, central, variations):
        systs = [(x, *y) for x, y in variations.items()]
        name, up, down = list(map(list, zip(*systs)))
        self.weights[sector_id].add_multivariation(weight_name, central, name, up, down)

    def weight(self, sector_id, modifier):
        k = (sector_id, modifier)
        if k in self.__cache:
            return self.__cache[k]
        else:
            weights = self.weights[sample_id].weight(modifier)
            self.__cache[k] = weights
            return weights


def getSectionHash(am_descs):
    to_hash = [{"name": x.name, "configuration": x.configuration} for x in am_descs]
    return hash(json.dumps(to_hash, sort_keys=True))


@dataclass
class Sector:
    region_name: str
    sample_id: SampleId

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
        return SectorId(self.sample_id, self.region_name)

    def getSelectionId(self):
        return getSectionHash(self.selection)

    def getPreselectionId(self):
        return getSectionHash(self.selection)

    @staticmethod
    def fromRegion(region_desc, sample, module_repo):
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

        objects = resolveModules(region_desc.objects, ModuleType.Producer)
        weights = resolveModules(region_desc.weights, ModuleType.Weight)

        preselection_histograms = resolveModules(
            region_desc.preselection_histograms, ModuleType.Histogram
        )
        postselection_histograms = resolveModules(
            region_desc.histograms, ModuleType.Histogram
        )

        return Sector(
            region_name=region_desc.name,
            sample_id=sample.sample_id,
            description=region_desc.description,
            forbid_data=region_desc.forbid_data,
            preselection=preselection,
            selection=selection,
            objects=objects,
            # preselection_histograms=preselection_histograms,
            histograms=postselection_histograms,
            weights=weights,
        )


Chunk = namedtuple("Chunk", "file start end")


def getParamsForSector(sector_id, dataset_repo, era_repo):
    sample_id = sector_id.sample_id
    dataset = dataset_repo[sample_id.dataset_name]
    sample = dataset.getSample(sample_id.sample_name)
    era = era_repo[sample.era]
    return {**era.params, **sample.params}


@dataclass
class Category:
    name: str
    axis: Any
    values: Any


@dataclass
class SectorResult:
    histograms: dict[str, HistogramCollection]
    other_data: dict[str, Any]
    cutflow_data: Optional[Any]


@dataclass
class AnalysisResult:
    datasets_preprocessed: dict[str, "DatasetPreprocessed"]
    processed_chunks: dict[str, set[Chunk]]
    description: AnalysisDescription
    results: dict[tuple[str, str], SectorResult]


@dataclass
class SectorAnalyzer:
    _analyzer: "Analyzer" = field(repr=False)
    sector: Sector
    params: dict[str, Any]
    result: "SectorResult"
    categories: dict[str, Category] = field(default_factory=dict)

    @property
    def sector_id(self):
        return self.sector.sector_id

    @property
    def events(self):
        return self._analyzer.sample_events[self.sector_id]

    def makeHistogram(self, spec: HistogramSpec, values):
        """Add a histogram based on a specification and the values to fill it."""
        pass

    def addHistogram(self, hist):
        """Add an existing histogram"""
        pass

    def addCategory(self, category):
        self.categories[category.name] = category

    def addSelection(self, name, mask, type="and", stage="preselection"):
        self._analyzer.selection_manager.register(
            self.sector_id, name, mask, type=type, stage=stage
        )

    def addWeight(self, name, central, variations=None):
        varia = variations or {}
        self._analyzer.weight_manager.add(self.sector_id, name, central, varia)

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
            AnalysisStage.Preselection: SectorAnalyzer.Selector(
                self, AnalysisStage.Preselection
            ),
            AnalysisStage.Selection: SectorAnalyzer.Selector(
                self, AnalysisStage.Selection
            ),
            AnalysisStage.Categorization: SectorAnalyzer.Categorizer(self),
            AnalysisStage.Weights: SectorAnalyzer.Weighter(self),
            AnalysisStage.Histogramming: SectorAnalyzer.Histogrammer(self),
        }
        return mapping.get(stage)

    def applyToEvents(self, events, stage):
        sp = self.__getStageProcessor(stage)
        mapping = {
            AnalysisStage.Preselection: self.sector.preselection,
            AnalysisStage.Selection: self.sector.selection,
            AnalysisStage.Categorization: self.sector.categories,
            AnalysisStage.Histogramming: self.sector.histograms,
            AnalysisStage.ObjectDefinition: self.sector.objects,
            AnalysisStage.Weights: self.sector.weights,
        }

        for module in mapping[stage]:
            if sp:
                module(events, self.params, sp)
            else:
                module(events, self.params)


@dataclass
class Analyzer:
    description: AnalysisDescription

    datasets_preprocessed: dict[str, "DatasetPreprocessed"] = field(
        default_factory=dict
    )
    sample_events: dict[SampleId, Any] = field(default_factory=dict)
    sample_reports: dict[SampleId, Any] = field(default_factory=dict)
    preselected_events: dict[SampleId, Any] = field(default_factory=dict)
    sector_events: dict[SectorId, Any] = field(default_factory=dict)

    weight_manager: WeightManager = field(default_factory=WeightManager)
    selection_manager: SelectionManager = field(default_factory=SelectionManager)

    sample_repo: "SampleRepo" = None
    era_repo: "EraRepo" = None

    def getSectorAnalyzers(self):
        ret = []
        sector_pairs = self.description.getSectors()
        for sample_name, region_name in sector_pairs:
            dataset = self.sample_repo[sample_name]
            region = self.description.getRegion(region_name)
            for sample in dataset.samples:
                sector = Sector.fromRegion(region, sample, MODULE_REPO)
                params = getParamsForSector(
                    sector.sector_id, self.sample_repo, self.era_repo
                )
                sa = SectorAnalyzer(self, sector, params, None)
                ret.append(sa)
        return ret

    def getEvents(self, sample_id, stage=None):
        return self.sample_events[sample_id]


# @singledispatch
def getEvents(arg, known_form=None, cache=None):
    from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory

    events, report = NanoEventsFactory.from_root(
        arg,
        schemaclass=NanoAODSchema,
        uproot_options=dict(
            allow_read_errors_with_report=True,
            timeout=30,
        ),
        known_base_form=known_form,
        persistent_cache=cache,
    ).events()
    return events, report


def runAnalysis(analyzer):

    sector_analyzers = analyzer.getSectorAnalyzers()
    print(sector_analyzers)

    for d in analyzer.datasets_preprocessed:
        analyzer.sample_events[d.sample_id] = getEvents(d)

    for sa in sector_analyzers:
        events = analyzer.getEvents(sa.sector.sample_id)
        sa.applyToEvents(events, AnalysisStage.Preselection)

    for sample_id, events in analyzer.sample_events.items():
        # events = analyzer.getEvents(sa.sector.sample_id)
        analyzer.preselected_events[
            sample_id
        ] = analyzer.selection_manager.maskPreselection(sample_id, events)

    for sa in sector_analyzers:
        events = analyzer.preselected_events[sa.sector.sample_id]
        sa.applyToEvents(events, AnalysisStage.ObjectDefinition)

    for sa in sector_analyzers:
        events = analyzer.preselected_events[sa.sector.sample_id]
        sa.applyToEvents(events, AnalysisStage.Selection)

    a.selection_manager.addPreselectionMasks()
    print(analyzer.selection_manager)

    for sa in sector_analyzers:

        events = analyzer.preselected_events[sa.sector.sample_id]
        analyzer.sector_events[sa.sector_id] = analyzer.selection_manager.maskSector(
            sa.sector_id, events
        )

    for sa in sector_analyzers:
        events = analyzer.sector_events[sa.sector_id]
        sa.applyToEvents(events, AnalysisStage.Weights)

    print(analyzer.weight_manager)

    for sa in sector_analyzers:
        events = analyzer.sector_events[sa.sector_id]
        sa.applyToEvents(events, AnalysisStage.Categorization)

    for sa in sector_analyzers:
        events = analyzer.sector_events[sa.sector_id]
        sa.applyToEvents(events, AnalysisStage.Histogramming)


class FakeRepo:
    def create(self, type, name, **kwargs):
        return dict(type=type, name=name, **kwargs)


def runAnalysisDescription(desc):

    analyzer = Analyzer(desc)
    sas = []

    # for n,d in analyzer.datasets_preprocessed.items():
    #    analyzer.sample_events[n] = getEvents(d)

    for sample_id, region_name in desc.getSectors():
        region = desc.getRegion(region_name)
        sec = region.getSector(sample_id, FakeRepo())
        sas.append(SectorAnalyzer(analyzer, sample_id, sec))

    for s in sas:
        print("=========================")
        print(s.region)
        for p in s.region.preselection:
            print(f"Processing preselection {p}")

        for p in s.region.objects:
            print(f"Processing preselection {p}")


if __name__ == "__main__":
    import analyzer.modules
    from analyzer.datasets import DatasetRepo, EraRepo

    d = yaml.safe_load(open("pydtest.yaml", "r"))

    an = AnalysisDescription(**d)
    print(an)

    em = EraRepo()
    em.load("analyzer_resources/eras")

    dm = DatasetRepo()
    dm.load("analyzer_resources/datasets")

    r = preprocessBulk(
        dm,
        [
            SampleId("signal_312_2000_1900", "signal_312_2000_1900"),
            SampleId("signal_312_1500_1400", "signal_312_1500_1400"),
        ],
        file_retrieval_kwargs=dict(require_location="eos"),
    )
    a = Analyzer(an)
    a.sample_repo = dm
    a.era_repo = em

    ss = a.getSectorAnalyzers()

    for x in r:
        ds = x.getCoffeaDataset(dm, require_location="eos")
        print(ds)
        e = getEvents(ds["files"])
        a.sample_events[x.sample_id] = e[0]
        a.sample_reports[x.sample_id] = e[1]

    print(a.sample_events)
    runAnalysis(a)
    for s, e in a.sector_events.items():
        e.visualize(filename="images/{s}.png")

#
#    print(ss)
