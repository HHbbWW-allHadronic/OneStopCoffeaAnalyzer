# from __future__ import annotations
import enum
import logging
from typing import (
    Any,
    ClassVar,
    Optional,
    Union,
)

import yaml
from pydantic import (
    BaseModel,
    Field,
)
from rich import print

logger = logging.getLogger(__name__)


class AnalysisStage(str, enum.Enum):
    Preselection = "Preselection"
    ObjectDefinition = "ObjectDefinition"
    Selection = "Selection"
    Weights = "Weights"
    Categorization = "Categorization"
    Histogramming = "Histogramming"


class HistogramSpec(BaseModel):
    name: str
    axes: list[Any]
    storage: str = "weight"
    description: str
    weights: list[str]
    variations: list[str]
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


class AnalysisDescription(BaseModel):
    name: str
    samples: dict[str, Union[list[str], str]]
    regions: list[RegionDescription]
    general_config: dict[str, Any] = Field(default_factory=dict)
    special_region_names: ClassVar[tuple[str]] = ("All",)

    def getRegion(self, name):
        try:
            return next(x for x in self.regions if x.name == name)
        except StopIteration as e:
            raise KeyError(f'No region "{name}"')


def main():
    d = yaml.safe_load(open("pydtest.yaml", "r"))
    print(d)
    an = AnalysisDescription(**d)
    print(an)


if __name__ == "__main__":
    main()
