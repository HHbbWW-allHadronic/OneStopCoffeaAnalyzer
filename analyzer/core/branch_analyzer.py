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
import distributed
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

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track

logger = logging.getLogger("analyzer.core")


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
            n_minus_one=self.n_minus_one + other.n_minus_one,
        )

    @property
    def selection_efficiency(self):
        return self.cutflow[-1][1] / self.cutflow[0][1]


@dataclass
class SelectionSet:
    """
    Selection for a single sample.
    Stores the preselection and selection masks.
    The selection mask is relative to the "or" of all the preselection cuts.
    """

    selection: PackedSelection = field(default_factory=PackedSelection)

    parent_names: Optional[list[str]] = None
    parent: Optional["SelectionSet"] = None

    def __eq__(self, other):
        return (
            self.parent == other.parent
            and self.selection.names == other.selection.names
            and self.parent_names == other.parent_names
        )

    def allNames(self):
        ret = self.selection.names
        if parent is not None:
            return ret + self.parent.names

    def addMask(self, stage, name, mask):
        if not name in self.allNames():
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
        ret = Cutflow(cutflow=cumcuts, one_cut=onecut, n_minus_one=nmocuts)
        if parent is not None:
            parent_cutflow = parent.getCutflow(self.parent_names)
            ret = parent_cutflow + ret

        return ret


@dataclass
class Selection:
    select_from: SelectionSet
    names: tuple[str] = field(default_factory=tuple)

    def __add__(self, name):
        self.names = self.names + (name,)

    def __eq__(self, other):
        return self.select_from == other.select_from and self.names == other.names



SHAPE_VAR_SEPARATOR="__"

@dataclass
class Column:
    name: str
    shape_variations: list[ str] = field(default_factory=list)


    def getColumnName(shape_var=None):
        if shape_var is None:
            return self.name
        return name + SHAPE_VAR_SEPARATOR + shape_var

@dataclass
class Columns:
    events: Any
    columns: dict[str, Column] = field(default_factory=dict)

    def allShapes(self):
        return list(it.chain.from_iterable(x.shape_variations for x in self.columns))

    def add(self, name, nominal_value, variations=None):
        col = Column(
            name=name, nominal_value=nominal_value, shape_variations=list(variations)
        )
        self.events[col.name] = nominal_value
        for svn, val in variations.items():
            self.events[col.getColumnName(svn)]  = val

    def get(self, name, variation=None):
        col = self.columns[name]
        return self.events[col.getColumnName(variation)]

    def __iter__(self):
        return iter(events.fields)


class ColumnShapeSyst(Columns):
    def __init__(self, base, syst=None):
        self.base = base
        self.this = Column(self.base.events)
        self.syst = syst

    def add(self, name, nominal_value, variations=None, depends_col=None):
        if depends_col is not None and depends_shape == self.syst[0]:
            target = self.this
        else:
            target = self.base

        if self.syst is None:
            return target.add(name,nominal_value, variations=variations)
        else:
            return target.add(name,nominal_value, variations=None)

    def get(self, name):
        if name == syst[0]:
            s = self.syst[1]
        else:
            s = None
        if name in self.this:
            return self.this.get(name, self.syst[1])
        else:
            return self.base.get(name, self.syst[1])


    def __iter__(self):
        return it.chain(iter(self.this), iter(self.base))





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

        selector = Selector(selection, selection_set)
        for module in self.preselection:
            module(events, params, selector)
        return selection

    def runCorrections(self, events, params, columns=None):
        if columns is None:
            columns = Columns(events)
        for module in self.corrections:
            module(events, params, columns)
        return columns


    def runBranch(self, columns, params, variation=None):
        runObject
            



    def runObject(self, columns, params, variation=None):
        shape_columns = ColumnShapeSyst(columns, variation=variation)
        for module in self.corrections:
            module(events, params, shape_columns)
        return columns

    def run(self, columns, params, variation=None):
        shape_columns = ColumnShapeSyst(columns, variation=variation)
        for module in self.corrections:
            module(events, params, shape_columns)
        return shape_columns



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



    def runBranch(self, events, params, columns, variation=None):
        pass
        

        

    def runPreselectionGroup(self, events, params, region_analyzers, preselection, preselection_set):
        mask = preselection_set.getMask(preselection.names)
        events = events[mask]
        columns = Columns(events)

        for ra in region_analyzers:
            ra.runCorrections(events,params,columns)

        shape_systs = columns.
        return columns

    def run(self, events, params):
        preselection_set: SelectionSet = field(default_factory=SelectionSet)
        region_preselections = []
        for analyzer in self.region_analyzers:
            region_preselections.append((region.name, analyzer.runPreselection(events, params, self.preselection_set)))
        k = key=lambda x: x[0].names
        presel_regions = it.groupby(sorted(region_preselections, key=k),key=k)
        ret = {}
        for _, items in presel_regions:
            items = list(items)
            sel = items[0][1]
            ret.update(self.runPreselectionGroup(events, params, [x[1] for x in items], sel, preselection_set))
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
