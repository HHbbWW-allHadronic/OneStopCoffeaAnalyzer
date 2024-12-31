import itertools as it


from analyzer.configuration import CONFIG
from pydantic import BaseModel, Field
import awkward as ak

from .analysis_modules import (
    ModuleType,
    ConfiguredAnalyzerModule,
)
from .histograms import Histogrammer
from .specifiers import SectorParams, SubSectorParams
from .columns import Columns
from .selection import Selection, SelectionSet, Selector
from .weights import Weighter
import logging

if CONFIG.PRETTY_MODE:
    pass

logger = logging.getLogger(__name__)


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

    def ensureFunction(self, module_repo):
        todo = [
            preselection,
            corrections,
            objects,
            selection,
            categories,
            histograms,
            weights,
        ]
        for item in todo:
            for module in item:
                if module.module._function is None:
                    module.module._function = module_repo.getFunction(
                        module.module.type, module.module.name
                    )

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

        selector = Selector(selection, selection_set)
        for module in self.preselection:
            module(events, params, selector)
        return selection

    def runSelection(self, columns, params, selection_set=None):
        params = self.getSectorParams(params)
        if selection_set is None:
            selection_set = SelectionSet()
        selection = Selection(select_from=selection_set)
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

    def runPostSelection(self, columns, params, histogram_storage):
        params = self.getSectorParams(params)
        active_shape = columns.syst
        if columns.delayed:
            size = None
        else:
            size = ak.num(columns.events, axis=0)
        weighter = Weighter(size=size, ignore_systematics=active_shape is not None)

        categories = []
        for module in self.weights:
            module(columns, params, weighter)
        for module in categories:
            module(columns, params)
        histogrammer = Histogrammer(
            storage=histogram_storage,
            weighter=weighter,
            categories=categories,
            active_shape_systematic=active_shape,
            delayed=columns.delayed,
        )
        for module in self.histograms:
            module(columns, params, histogrammer)


__subsector_param_cache = {}
__sample_param_cache = {}


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


def getParamsSample(sample_id, dataset_repo, era_repo):
    if sample_id in __sample_param_cache:
        return __subsector_param_cache[subsector_id]

    dataset = dataset_repo[sample_id.dataset_name]
    params = dataset.getSample(sample_id.sample_name).params
    params.dataset.populateEra(era_repo)
    return params
