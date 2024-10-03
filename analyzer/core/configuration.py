# from __future__ import annotations
import enum
import logging
from typing import Any, ClassVar, Optional, Union

from pydantic import BaseModel, Field
from .specifiers import SampleSpec

logger = logging.getLogger(__name__)

class AnalysisStage(str, enum.Enum):
    Preselection = "Preselection"
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
    cluster_type: str
    max_workers: int
    chunk_size: int = 100000
    worker_memory: Optional[str] = "4GB"
    dashboard_address: Optional[str] = None
    schedd_address: Optional[str] = None
    worker_timeout: int = 3600


class FileConfig(BaseModel):
    location_priority_regex: list[str] = [".*FNAL.*", ".*US.*", ".*(DE|IT|CH|FR).*", ".*(T0|T1|T2).*","eos"]
    use_replicas: bool = True


class AnalysisDescription(BaseModel):
    name: str
    execution_config: ExecutionConfig
    file_config: FileConfig
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
        


