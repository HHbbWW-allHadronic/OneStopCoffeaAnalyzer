import functools as ft
import itertools as it
from collections import ChainMap
from dataclasses import dataclass, field
from typing import Any

import pydantic as pyd
from analyzer.datasets import SampleId

from .analysis_modules import AnalyzerModule, ModuleType
from .exceptions import AnalysisConfigurationError


# SectorId= namedtuple("SectorId", "sample_id region_name")
@pyd.dataclasses.dataclass(frozen=True)
class SectorId:
    sample_id: SampleId
    region_name: str

    @pyd.model_serializer
    def serialize(self) -> str:
        return self.region_name + "___" + self.sample_id.serialize()

    @pyd.model_validator(mode="before")
    @classmethod
    def isStr(self, value):
        if isinstance(value, str):
            a, *b = value.split("___")
            return {"region_name": a, "sample_id": "___".join(b)}
        else:
            return value


@dataclass
class Sector:
    sector_id: SectorId

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
        return self.sector_id.sample_id

    @property
    def region_name(self):
        return self.sector_id.region_name

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
            sector_id=SectorId(sample.sample_id, region_desc.name),
            description=region_desc.description,
            forbid_data=region_desc.forbid_data,
            preselection=preselection,
            selection=selection,
            objects=objects,
            # preselection_histograms=preselection_histograms,
            histograms=postselection_histograms,
            weights=weights,
        )


@dataclass
class SectorParams:
    era_params: dict[str, Any]
    dataset_params: dict[str, Any]
    sample_params: dict[str, Any]
    sector_id: SectorId

    @ft.cached_property
    def _all_params(self):
        def notNone(m):
            return {x: y for x, y in m.items() if y is not None}

        ret = ChainMap(
            notNone(self.sample_params),
            notNone(self.dataset_params),
            notNone(self.era_params),
        )
        return ret

    def __getitem__(self, key):
        if key == "sector_id":
            return self.sector_id
        return self._all_params[key]

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError
        return self[attr]


__sector_param_cache ={}
def getParamsForSector(sector_id, dataset_repo, era_repo):
    if sector_id in __sector_param_cache:
        return __sector_param_cache[sector_id]

    sample_id = sector_id.sample_id
    dataset = dataset_repo[sample_id.dataset_name]
    sample = dataset.getSample(sample_id.sample_name)
    sp = sample.params
    era = era_repo[sample.era]
    p =  SectorParams(
        era_params=era.params,
        sample_params=sp["sample_params"],
        dataset_params=sp["dataset_params"],
        sector_id=sector_id,
    )
    __sector_param_cache[sector_id] = p
    return p
