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

from .analysis_modules import MODULE_REPO, AnalyzerModule, ModuleType
from .common_types import Scalar
from .specifiers import SampleSpec, SubSectorId
from .columns import Column,Columns, ColumnShapeSyst
from .selection import Cutflow,Selection,SelectionSet

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track

logger = logging.getLogger("analyzer.core")


@dataclass
class RegionAnalyzer:
    region_name: str
    description: str = ""
    forbid_data: bool = False

    preselection: list[AnalyzerModule] = field(default_factory=list)
    corrections: list[AnalyzerModule] = field(default_factory=list)

    objects: list[AnalyzerModule] = field(default_factory=list)
    selection: list[AnalyzerModule] = field(default_factory=list)
    categories: list[AnalyzerModule] = field(default_factory=list)
    histograms: list[AnalyzerModule] = field(default_factory=list)
    weights: list[AnalyzerModule] = field(default_factory=list)

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

    def runPreselection(self, events, params, selection_set=None):

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

        print(selection_set)
        selector = Selector(selection, selection_set)
        for module in self.preselection:
            print(module)
            print(selector)
            module(events, params, selector)
        return selection

    def runCorrections(self, events, params, columns=None):
        if columns is None:
            columns = Columns(events)
        for module in self.corrections:
            module(columns, params)
        return columns


    def runBranch(self, columns, params, variation=None):
        columns = ColumnShapeSyst(columns, syst=variation)
        runObject(columns, params, variation=None)
            



    def runObject(self, columns, params, variation=None):
        for module in self.corrections:
            module(events, params)
        return columns

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



    def runBranch(self, region_analyzers, columns, params, variation=None):
        logger.info(f'Running analysis branch for variation {variation}')
        for ra in region_analyzers:
            ra.runBranch(columns, params, variation=variation)

        

        

    def runPreselectionGroup(self, events, params, region_analyzers, preselection, preselection_set):
        mask = preselection_set.getMask(preselection.names)
        events = events[mask]
        columns = Columns(events)

        for ra in region_analyzers:
            ra.runCorrections(events,params,columns)

        branches = [None] + columns.allShapes()
        for variation in branches:
            self.runBranch(region_analyzers, events, params, variation=variation)

    def run(self, events, params):
        preselection_set=  SelectionSet()
        region_preselections = []
        for analyzer in self.region_analyzers:
            region_preselections.append((analyzer, analyzer.runPreselection(events, params, preselection_set)))
        k = lambda x: x[1].names
        print(region_preselections)

        presel_regions = it.groupby(sorted(region_preselections, key=k),key=k)
        presel_regions = {x:list(y) for x,y in presel_regions}
        ret = {}
        for presels, items in presel_regions.items():
            logger.info(f'Running over preselection region "{presels}" containing "{len(items)}" regions.')
            sel = items[0][1]
            ret.update(self.runPreselectionGroup(events, params, [x[0] for x in items], sel, preselection_set))
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
        logger.debug(f'Getting analyzer for dataset "{dataset_name}" and region "{region_name}"')
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
    # print(region)
    # s = dr["signal_312_2000_1900"].getSample("signal_312_2000_1900")
    # ss = RegionAnalyzer.fromRegion(region, s, MODULE_REPO, er)
    NanoAODSchema.warn_missing_crossrefs = False
    fname = "https://raw.githubusercontent.com/CoffeaTeam/coffea/master/tests/samples/nano_dy.root"
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
    print(sample_params)
    analyzer = Analyzer(one_sample)
    analyzer.run(events, sample_params)
