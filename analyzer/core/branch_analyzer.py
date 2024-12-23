import concurrent.futures
import copy
import enum
import inspect
import itertools as it
import logging
import pickle as pkl
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import yaml

import awkward as ak
import dask
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId, SampleType
from analyzer.utils.file_tools import extractCmsLocation
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from pydantic import BaseModel, ConfigDict, Field

from .analysis_modules import (
    MODULE_REPO,
    AnalyzerModule,
    ModuleType,
    ConfiguredAnalyzerModule,
)
from .common_types import Scalar
from .histograms import HistogramSpec, HistogramCollection
from .specifiers import SampleSpec, SubSectorId, SectorParams, SubSectorParams
from .columns import Column, Columns
from .selection import Cutflow, Selection, SelectionSet
import analyzer.core.results as results

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track

logger = logging.getLogger("analyzer.core")


@dataclass
class Category:
    name: str
    axis: Any
    values: Any
    distinct_values: set[Union[int, str, float]] = field(default_factory=set)


class Histogrammer:
    def __init__(self, weighter, categories=None, active_shape_systematic=None):
        self.weighter = weighter
        self.categories = categories
        self.active_shape_systematic = active_shape_systematic
        self.results = {}

    def H(
        self,
        name,
        axes,
        values,
        variations=None,
        weights=None,
        description="",
        no_scale=False,
        mask=None,
        storage="weight",
    ):
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        if variations is None:
            variations = self.weighter.variations
        spec = HistogramSpec(
            name=name,
            axes=axes,
            storage=storage,
            description=description,
            weights=weights,
            variations=variations,
            no_scale=no_scale,
        )
        ret = HistogramCollection.create(
            spec,
            values,
            self.categories,
            self.weighter,
            active_shape_systematic=self.active_shape_systematic,
            mask=mask,
        )
        self.results[spec.name] = ret
        return ret


class Weighter:
    def __init__(self, ignore_systematics=False):
        self.weights = Weights(None, storeIndividual=True)
        self.ignore_systematics = ignore_systematics
        self.__cache = {}

    def add(self, weight_name, central, variations=None):
        if variations and not self.ignore_systematics:
            systs = [(x, *y) for x, y in variations.items()]
            name, up, down = list(map(list, zip(*systs)))
            self.weights.add_multivariation(weight_name, central, name, up, down)
        else:
            self.weights.add(weight_name, central)

    @property
    def variations(self):
        return list(self.weights.variations)

    @property
    def weight_names(self):
        return list(self.weights._weights)

    @property
    def total_weight(self):
        return (
            ak.sum(self.weight(), axis=0),
            ak.sum(self.weight() ** 2, axis=0),
        )

    def weight(self, modifier=None, include=None, exclude=None):
        inc = include or []
        exc = exclude or []
        k = (modifier, tuple(inc), tuple(exc))
        if k in self.__cache:
            return self.__cache[k]
        if include or exclude:
            ret = self.weights.partial_weight(
                modifier=modifier, include=inc, exclude=exc
            )
        else:
            ret = self.weights.weight(modifier)
        self.__cache[k] = ret
        return ret


class RegionAnalyzer(BaseModel):
    region_name: str
    description: str = ""
    forbid_data: bool = False

    preselection: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    corrections: list[ConfiguredAnalyzerModule] = Field(default_factory=list)

    objects: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    selection: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    categories: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    histograms: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    weights: list[ConfiguredAnalyzerModule] = Field(default_factory=list)

    @staticmethod
    def fromRegion(region_desc, sample, module_repo, era_repo):
        name = region_desc.name
        sample_params = sample.params
        dataset_params = sample_params.dataset
        dataset_params.populateEra(era_repo)
        if region_desc.forbid_data and dataset_params.sample_type == "Data":
            raise AnalysisConfigurationError(
                f"Region '{region_desc.name}' is marked with 'forbid_data'"
                f"but is recieving Data sample '{sample.name}'"
            )

        def resolveModules(l, t):
            ret = [
                mod
                for mod in l
                if not mod.sample_spec or mod.sample_spec.passes(dataset_params)
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
        corrections = resolveModules(region_desc.corrections, ModuleType.Producer)
        objects = resolveModules(region_desc.objects, ModuleType.Producer)

        selection = resolveModules(region_desc.selection, ModuleType.Selection)

        weights = resolveModules(region_desc.weights, ModuleType.Weight)
        categories = resolveModules(region_desc.categories, ModuleType.Categorization)

        preselection_histograms = resolveModules(
            region_desc.preselection_histograms, ModuleType.Histogram
        )
        postselection_histograms = resolveModules(
            region_desc.histograms, ModuleType.Histogram
        )

        return RegionAnalyzer(
            region_name=name,
            description=region_desc.description,
            forbid_data=region_desc.forbid_data,
            preselection=preselection,
            objects=objects,
            corrections=corrections,
            selection=selection,
            histograms=postselection_histograms,
            categories=categories,
            weights=weights,
        )

    def getSectorParams(self, sample_params):
        return SubSectorParams(sample=sample_params, region_name=self.region_name)

    def runPreselection(self, events, params, selection_set=None):
        params = self.getSectorParams(params)

        if selection_set is None:
            selection_set = SelectionSet()
        selection = Selection(select_from=selection_set)

        class Selector:
            def __init__(self, selection, selection_set):
                self.selection = selection
                self.selection_set = selection_set

            def add(self, name, mask):
                self.selection += name
                return self.selection_set.addMask(name, mask)

        selector = Selector(selection, selection_set)
        for module in self.preselection:
            module(events, params, selector)
        return selection

    def runSelection(self, columns, params, selection_set=None):
        params = self.getSectorParams(params)
        if selection_set is None:
            selection_set = SelectionSet()
        selection = Selection(select_from=selection_set)

        class Selector:
            def __init__(self, selection, selection_set):
                self.selection = selection
                self.selection_set = selection_set

            def add(self, name, mask):
                self.selection += name
                return self.selection_set.addMask(name, mask)

        selector = Selector(selection, selection_set)
        for module in self.selection:
            module(columns, params, selector)
        return selection

    def runCorrections(self, events, params, columns=None):
        params = self.getSectorParams(params)
        if columns is None:
            columns = Columns(events)
        for module in self.corrections:
            module(columns, params)
        return columns

    def runObjects(self, columns, params):
        params = self.getSectorParams(params)
        for module in self.objects:
            module(columns, params)
        return columns

    def runPostSelection(self, columns, params):
        params = self.getSectorParams(params)
        active_shape = columns.syst
        weighter = Weighter(ignore_systematics=active_shape is not None)

        categories = []
        for module in self.weights:
            module(columns, params, weighter)
        for module in categories:
            module(columns, params)
        histogrammer = Histogrammer(
            weighter=weighter,
            categories=categories,
            active_shape_systematic=active_shape,
        )
        for module in self.histograms:
            module(columns, params, histogrammer)


        print(isinstance(self, RegionAnalyzer))
        return results.SubSectorResult(
            region=self.model_dump(),
            params=params,
            histograms=histogrammer.results,
            other_data={},
            cutflow_data=None,
        )

    # def run(self, columns, params, variation=None):
    #     shape_columns = ColumnShapeSyst(columns, variation=variation)
    #     for module in self.corrections:
    #         module(events, params, shape_columns)
    #     return shape_columns


__subsector_param_cache = {}


def getParamsForSubSector(subsector_id, dataset_repo, era_repo):
    if subsector_id in __subsector_param_cache:
        return __subsector_param_cache[subsector_id]

    sample_id = subsector_id.sample_id
    dataset = dataset_repo[sample_id.dataset_name]
    params = dataset.getSample(sample_id.sample_name).params
    dataset_params = params.dataset
    dataset_params.populateEra(era_repo)
    sector_params = SectorParams(
        dataset=dataset_params,
        region={"region_name": subsector_id.region_name},
    )
    p = SubSectorParams(
        sector=sector_params,
        sample=params.sample,
        subsector_id=subsector_id,
    )
    __subsector_param_cache[subsector_id] = p
    return p


__sample_param_cache = {}


def getParamsSample(sample_id, dataset_repo, era_repo):
    if sample_id in __sample_param_cache:
        return __subsector_param_cache[subsector_id]

    dataset = dataset_repo[sample_id.dataset_name]
    params = dataset.getSample(sample_id.sample_name).params
    params.dataset.populateEra(era_repo)
    return params


@dataclass
class Analyzer:
    region_analyzers: list[RegionAnalyzer]

    def runBranch(
        self, region_analyzers, columns, params, preselection, variation=None
    ):
        logger.info(f"Running analysis branch for variation {variation}")
        selection_set = SelectionSet()
        columns = columns.withSyst(variation)
        for ra in region_analyzers:
            ra.runObjects(columns, params)
        ret = {}
        for ra in region_analyzers:
            selection = ra.runSelection(columns, params, selection_set)
            mask = selection.getMask()
            print(columns)
            new_cols = columns.withEvents(columns.events[mask])
            res = ra.runPostSelection(new_cols, params)
            ret[variation](res)
        return ret

    def runPreselectionGroup(
        self, events, params, region_analyzers, preselection, preselection_set
    ):
        mask = preselection.getMask()
        print(mask)
        events = events[mask]
        columns = Columns(events)

        for ra in region_analyzers:
            ra.runCorrections(events, params, columns)

        branches = [None] + columns.allShapes()
        logger.info(f"Known variations are {branches}")
        ret = []
        for variation in branches:
            logger.info(f'Running branch for variation "{variation}"')
            res = self.runBranch(
                region_analyzers, columns, params, preselection, variation=variation
            )
            ret.append(res)
        return list(it.chain.from_iterable(ret))

    def run(self, events, params):
        preselection_set = SelectionSet()
        region_preselections = []
        for analyzer in self.region_analyzers:
            region_preselections.append(
                (analyzer, analyzer.runPreselection(events, params, preselection_set))
            )
        k = lambda x: x[1].names

        presel_regions = it.groupby(sorted(region_preselections, key=k), key=k)
        presel_regions = {x: list(y) for x, y in presel_regions}
        ret = []
        for presels, items in presel_regions.items():
            logger.info(
                f'Running over preselection region "{presels}" containing "{len(items)}" regions.'
            )
            sel = items[0][1]
            ret += self.runPreselectionGroup(
                events, params, [x[0] for x in items], sel, preselection_set
            )

        return ret


class AnalysisStage(str, enum.Enum):
    Preselection = "Preselection"
    Correction = "Correction"
    ObjectDefinition = "ObjectDefinition"
    Selection = "Selection"
    Weights = "Weights"
    Categorization = "Categorization"
    Histogramming = "Histogramming"


class ModuleDescription(BaseModel):
    name: str
    sample_spec: Optional[SampleSpec] = None
    config: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None


class RegionDescription(BaseModel):
    name: str
    use_region: bool = True
    forbid_data: bool = False
    description: str = ""

    selection: list[ModuleDescription] = Field(default_factory=list)
    objects: list[ModuleDescription] = Field(default_factory=list)
    corrections: list[ModuleDescription] = Field(default_factory=list)
    preselection: list[ModuleDescription] = Field(default_factory=list)
    preselection_histograms: list[ModuleDescription] = Field(default_factory=list)
    categories: list[ModuleDescription] = Field(default_factory=list)
    histograms: list[ModuleDescription] = Field(default_factory=list)
    weights: list[ModuleDescription] = Field(default_factory=list)


class ExecutionConfig(BaseModel):
    cluster_type: str = "local"
    max_workers: int = 20
    step_size: int = 100000
    worker_memory: Optional[str] = "4GB"
    dashboard_address: Optional[str] = None
    schedd_address: Optional[str] = None
    worker_timeout: int = 3600
    extra_files: Optional[list[str]] = None


class FileConfig(BaseModel):
    location_priority_regex: list[str] = [
        ".*FNAL.*",
        ".*US.*",
        ".*(DE|IT|CH|FR).*",
        ".*(T0|T1|T2).*",
        "eos",
    ]
    use_replicas: bool = True


class AnalysisDescription(BaseModel):
    name: str
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    file_config: FileConfig = Field(default_factory=FileConfig)
    samples: dict[str, Union[list[str], str]]
    regions: list[RegionDescription]
    general_config: dict[str, Any] = Field(default_factory=dict)
    special_region_names: ClassVar[tuple[str]] = ("All",)

    def getRegion(self, name):
        try:
            return next(x for x in self.regions if x.name == name)
        except StopIteration as e:
            raise KeyError(f'No region "{name}"')

    def __eq__(self, other):
        def asTuple(ad):
            return (ad.name, ad.samples, ad.regions)

        return asTuple(self) == asTuple(other)


def loadDescription(input_path):
    with open(input_path, "rb") as config_file:
        data = yaml.safe_load(config_file)
    return AnalysisDescription(**data)


def getSubSectors(description, dataset_repo, era_repo):
    s_pairs = []
    ret = defaultdict(list)
    for dataset_name, regions in description.samples.items():
        if isinstance(regions, str) and regions == "All":
            regions = [r.name for r in description.regions]
        for r in regions:
            s_pairs.append((dataset_name, r))
    for dataset_name, region_name in s_pairs:
        logger.debug(
            f'Getting analyzer for dataset "{dataset_name}" and region "{region_name}"'
        )
        dataset = dataset_repo[dataset_name]
        region = description.getRegion(region_name)
        for sample in dataset.samples:
            subsector = RegionAnalyzer.fromRegion(
                region,
                sample,
                MODULE_REPO,
                era_repo,
            )
            ret[sample.sample_id].append(subsector)

    return ret


if __name__ == "__main__":
    import analyzer.modules
    from analyzer.logging import setup_logging

    setup_logging()

    d = loadDescription("configurations/data_mc_comp.yaml")

    dr = DatasetRepo.getConfig()
    er = EraRepo.getConfig()

    # region = d.getRegion("Signal312")
    #
    # s = dr["signal_312_2000_1900"].getSample("signal_312_2000_1900")
    # ss = RegionAnalyzer.fromRegion(region, s, MODULE_REPO, er)
    NanoAODSchema.warn_missing_crossrefs = False
    fname = "https://raw.githubusercontent.com/CoffeaTeam/coffea/master/tests/samples/nano_dy.root"
    fname = "test.root"
    events = NanoEventsFactory.from_root(
        {fname: "Events"},
        schemaclass=NanoAODSchema,
        metadata={"dataset": "DYJets"},
    ).events()

    subsectors = getSubSectors(d, dr, er)
    one_sample = subsectors[
        SampleId(
            dataset_name="signal_312_2000_1900", sample_name="signal_312_2000_1900"
        )
    ]
    sample_params = getParamsSample(
        SampleId(
            dataset_name="signal_312_2000_1900", sample_name="signal_312_2000_1900"
        ),
        dr,
        er,
    )
    analyzer = Analyzer(one_sample)
    # e = events.compute()
    # print(e)
    r = analyzer.run(events, sample_params)
    print(r.model_dump())
