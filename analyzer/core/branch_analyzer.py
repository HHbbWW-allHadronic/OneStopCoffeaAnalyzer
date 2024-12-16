import concurrent.futures
import copy
import enum
import inspect
import itertools as it
import logging
from pydantic import BaseModel, ConfigDict
from .common_types import Scalar
from coffea.analysis_tools import PackedSelection
import pickle as pkl
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import awkward as ak
import dask
import distributed
import yaml
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId, SampleType
from analyzer.utils.file_tools import extractCmsLocation
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from pydantic import BaseModel, Field

from .analysis_modules import MODULE_REPO, AnalyzerModule, ModuleType
from .specifiers import SampleSpec, SubSectorId

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track


@dataclass
class Selection:
    or_names: tuple[str] = field(default_factory=tuple)
    and_names: tuple[str] = field(default_factory=tuple)

    def addOne(self, name, type="and"):
        if type == "and":
            s = Selection(or_names=tuple(), and_names=(name,))
        else:
            s = Selection(or_names=(name,), and_names=tuple())
        return self.merge(s)

    def merge(self, other):
        def addTuples(t, o):
            to_add = [x for x in o if x not in t]
            return tuple([*t, *to_add])

        return Selection(
            addTuples(self.or_names, other.or_names),
            addTuples(self.and_names, other.and_names),
        )


class Cutflow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cutflow: list[tuple[str, Scalar]]
    one_cut: list[tuple[str, Scalar]]
    n_minus_one: list[tuple[str, Scalar]]

    def __add__(self, other):
        def add_tuples(a, b):
            return [(x, y) for x, y in accumulate([dict(a), dict(b)]).items()]

        return Cutflow(
            cutflow=add_tuples(self.cutflow, other.cutflow),
            one_cut=add_tuples(self.one_cut, other.one_cut),
            n_minus_one=add_tuples(self.n_minus_one, other.n_minus_one),
        )

    def concat(self, other):
        return Cutflow(
            cutflow=self.cutflow + other.cutflow,
            one_cut=self.one_cut + other.one_cut,
            n_minus_one=self.n_minus_one+ other.n_minus_one,
        )

    @property
    def selection_efficiency(self):
        return self.cutflow[-1][1] / self.cutflow[0][1]


@dataclass
class Selection:
    """
    Selection for a single sample.
    Stores the preselection and selection masks.
    The selection mask is relative to the "or" of all the preselection cuts.
    """


    selection: PackedSelection

    parent_names: Optional[list[str]] = None
    parent: Optional[Selection] = None



    def allNames(self):
        ret =  self.selection.names 
        if parent is not None:
            return ret + self.parent.names

    def addMask(self, stage, name, mask):
        if name in self.allNames():
            raise KeyError(f'Selection name {name} already exists')

        logger.info(f'Adding name to selection stage "{stage}".')
        target.add(name, mask)


    def inclusiveMask(self):
        names = self.selection.names
        if not names:
            return None
        return self.selection.any(*names)

    def getMask(self, names):
        return self.selections.all(*names)

    def getCutflow(self, names):
        nmo = sel.nminusone(*names).result()
        cutflow = sel.cutflow(*names).result()
        onecut = list(map(tuple, zip(cutflow.labels, cutflow.nevonecut)))
        cumcuts = list(map(tuple, zip(cutflow.labels, cutflow.nevcutflow)))
        nmocuts = list(map(tuple, zip(nmo.labels, nmo.nev)))
        ret =  Cutflow(cutflow=cumcuts, one_cut=onecut, n_minus_one=nmocuts)
        if parent is not None:
            parent_cutflow = parent.getCutflow(self.parent_names)
            ret = parent_cutflow + ret

        return ret


@dataclass
class Column:
    name: str
    nominal_value: Any
    shape_variations: dict[str, Any] = field(default_factory=dict)


@dataclass
class Columns:
    columns: dict[str, Column] = field(default_factory=dict)

    def __getattr__(self, attr):
        return getattr(self.events, attr)


    def allShapes(self):
        return it.chain(x.shape_variations for x in self.columns)

    def add(self, name, nominal_value, variations=None):
        self.columns[name] = Column(
            name=name, nominal_value=nominal_value, shape_variations=variations
        )

    def get(self, name, variation=None):
        col = self.columns[name]
        if variation is None:
            return col.nominal_value
        else:
            return col.shape_variations[variation]


@dataclass
class SubSectorAnalyzer:
    description: str = ""
    forbid_data: bool = False

    preselection: list[AnalyzerModule] = field(default_factory=list)
    corrections: list[AnalyzerModule] = field(default_factory=list)

    objects: list[AnalyzerModule] = field(default_factory=list)
    selection: list[AnalyzerModule] = field(default_factory=list)
    categories: list[AnalyzerModule] = field(default_factory=list)
    histograms: list[AnalyzerModule] = field(default_factory=list)
    weights: list[AnalyzerModule] = field(default_factory=list)



    preselection: Selection
    selection: Selection
    columns: Columns
    weights: Weights

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
        categories = resolveModules(region_desc.categories, ModuleType.Categories)

        preselection_histograms = resolveModules(
            region_desc.preselection_histograms, ModuleType.Histogram
        )
        postselection_histograms = resolveModules(
            region_desc.histograms, ModuleType.Histogram
        )

        return SubSectorAnalyzer(
            subsector_id=SubSectorId(sample.sample_id, region_desc.name),
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

    def run(self, events):
        pass


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


@dataclass
class Analyzer:
    column_manager: Any
    subsector_analyzers: dict[str, SubSectorAnalyzer]

    def run(self, events):
        return {region_name: SubSectorResult}


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
    preselection: list[ModuleDescription] = Field(default_factory=list)
    preselection_histograms: list[ModuleDescription] = Field(default_factory=list)
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


if __name__ == "__main__":
    d = loadDescription("configurations/test_config.yaml")

    dr = DatasetRepo.getConfig()
    er = EraRepo.getConfig()

    region = d.getRegion("Signal312")
    print(region)
