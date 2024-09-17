# from __future__ import annotations
import copy
import enum
import inspect
import itertools as it
import json
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
from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    TypeAdapter,
    create_model,
    field_validator,
)
from pydantic_core import CoreSchema, core_schema
from rich import print

class HistogramSpec(BaseModel):
    name: str
    axes: list[Any]
    storage: str = "weight"
    description: str
    weights: list[str] = None
    weight_variations: dict[str, list[str]]
    no_scale: bool = False




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


class WeightDescription(BaseModel):
    name: str
    sector_spec: Optional[SectorSpec] = None
    variation: list[str] = Field(default_factory=list)
    region_config: Optional[dict[str, dict[str, Any]]] = None

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


class AnalysisDescription(BaseModel):
    name: str
    object_definitions: list[ModuleDescription]
    samples: dict[str, Union[list[str], str]]
    regions: list[RegionDescription]
    weights: list[WeightDescription]
    general_config: dict[str, Any] = Field(default_factory=dict)
    special_region_names: ClassVar[tuple[str]] = ("All",)

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



