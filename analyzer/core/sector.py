import itertools as it
from dataclasses import dataclass, field


from .analysis_modules import AnalyzerModule, ModuleType
from .exceptions import AnalysisConfigurationError
from .specifiers import SectorParams, SubSectorId, SubSectorParams


@dataclass
class SubSector:
    subsector_id: SubSectorId

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
    def sample_id(self):
        return self.subsector_id.sample_id

    @property
    def region_name(self):
        return self.subsector_id.region_name

    def getSelectionId(self):
        return getSectionHash(self.selection)

    def getPreselectionId(self):
        return getSectionHash(self.selection)

    @staticmethod
    def fromRegion(region_desc, sample, module_repo, era_repo):
        name = region_desc.name
        sample_params = sample.params
        dataset_params = sample_params.dataset_params
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
        selection = resolveModules(region_desc.selection, ModuleType.Selection)

        objects = resolveModules(region_desc.objects, ModuleType.Producer)
        weights = resolveModules(region_desc.weights, ModuleType.Weight)

        preselection_histograms = resolveModules(
            region_desc.preselection_histograms, ModuleType.Histogram
        )
        postselection_histograms = resolveModules(
            region_desc.histograms, ModuleType.Histogram
        )

        return SubSector(
            subsector_id=SubSectorId(sample.sample_id, region_desc.name),
            description=region_desc.description,
            forbid_data=region_desc.forbid_data,
            preselection=preselection,
            selection=selection,
            objects=objects,
            # preselection_histograms=preselection_histograms,
            histograms=postselection_histograms,
            weights=weights,
        )

__subsector_param_cache = {}

def getParamsForSubSector(subsector_id, dataset_repo, era_repo):
    if subsector_id in __subsector_param_cache:
        return __subsector_param_cache[subsector_id]

    sample_id = subsector_id.sample_id
    dataset = dataset_repo[sample_id.dataset_name]
    params = dataset.getSample(sample_id.sample_name).params
    dataset_params = params.dataset_params
    dataset_params.populateEra(era_repo)
    sector_params = SectorParams(
        dataset_params=dataset_params,
        region_params={"region_name": subsector_id.region_name},
    )
    p = SubSectorParams(
        sector_params=sector_params,
        sample_params=params.sample_params,
        subsector_id=subsector_id,
    )
    __subsector_param_cache[subsector_id] = p
    return p
