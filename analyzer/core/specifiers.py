import logging
from enum import Enum
from fnmatch import fnmatch
import re

import pydantic as pyd
from analyzer.datasets import DatasetParams, SampleId, SampleType, SampleParams
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
    TypeAdapter,
)

logger = logging.getLogger(__name__)


@pyd.dataclasses.dataclass(frozen=True)
class SubSectorId:
    sample_id: SampleId
    region_name: str

    @pyd.model_serializer
    def serialize(self) -> str:
        return self.region_name + "___" + self.sample_id.serialize()

    @pyd.model_validator(mode="before")
    @classmethod
    def isStr(self, value):
        if isinstance(value, str):
            a, b = value.split("___", 1)
            return {"region_name": a, "sample_id": b}
        else:
            return value


class PatternMode(str, Enum):
    REGEX = "REGEX"
    GLOB = "GLOB"


class Pattern(pyd.BaseModel):
    mode: PatternMode = PatternMode.GLOB
    pattern: str

    def match(self, string, *args, **kwargs):
        if self.mode == PatternMode.REGEX:
            ret = re.match(self.pattern, string, *args, **kwargs)
        else:
            ret = fnmatch(string, self.pattern)
        return ret

    @pyd.model_validator(mode="before")
    @classmethod
    def convertString(cls, data):
        if isinstance(data, str):
            if data.startswith("re:"):
                return {"mode": "REGEX", "pattern": data.removeprefix("re:")}
            elif data.startswith("glob:"):
                return {"mode": "GLOB", "pattern": data.removeprefix("glob:")}
            else:
                return {"mode": "GLOB", "pattern": data}
        else:
            return data


PatternList = TypeAdapter(list[Pattern])


@pyd.dataclasses.dataclass(frozen=True)
class ShapeVariationId:
    column_name: str
    shape_variation: str


class SectorParams(pyd.BaseModel):
    dataset: DatasetParams
    region_name: str


class SubSectorParams(pyd.BaseModel):
    sample: SampleParams
    region_name: str

    @property
    def dataset(self):
        return self.sample.dataset


class SampleSpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: list[Pattern] | Pattern | None = None
    era: list[Pattern] | Pattern | None = None
    sample_type: SampleType | None = None

    @field_validator("name", "era")
    @classmethod
    def makeList(cls, val, info):
        if val is None:
            return val
        if not isinstance(val, list):
            return [val]
        return val

    def passes(self, dataset_params, return_specificity=False, **kwargs):
        passes_names = not self.name or any(
            x.match(dataset_params.name) for x in self.name
        )
        passes_era = not self.era or any(
            x.match(dataset_params.era.name) for x in self.era
        )
        passes_sample_type = not self.sample_type or (
            dataset_params.sample_type == self.sample_type
        )
        passes = passes_names and passes_era and passes_sample_type
        if return_specificity:
            if not passes:
                return 0
            exact_name = any(dataset_params.name == x for x in self.name)
            exact_era = any(dataset_params.era == x for x in self.sample_era)
            return (
                100 * exact_name
                + 50 * (not exact_name and passes_names)
                + 30 * exact_era
                + 2 * (not exact_era and passes_era)
                + 5 * passes_sample_type
            )

        else:
            return passes


class SectorSpec(BaseModel):
    sample_spec: SampleSpec | None = None
    region_name: list[Pattern] | Pattern | None = None

    @field_validator("region_name")
    @classmethod
    def makeList(cls, val, info):
        if val is None:
            return val
        if not isinstance(val, list):
            return [val]
        return val

    def passes(self, sector_params, return_specificity=False):
        passes_sample = not self.sample_spec or self.sample_spec.passes(
            sector_params.dataset, return_specificity=return_specificity
        )
        passes_region = not self.region_name or any(
            x.match(sector_params.region_name) for x in self.region_name
        )
        passes = passes_sample and passes_region
        if return_specificity:
            exact_region = any(sector_params.region_name == x for x in self.region_name)
            return (
                80 * exact_region
                + 50 * (not exact_region and passes_region)
                + passes_sample
            )

        else:
            return passes

    @model_validator(mode="before")
    @classmethod
    def ifSingleton(cls, values):
        if "sample_spec" not in values:
            sample_spec = {
                "name": values.get("sample_name"),
                "era": values.get("era"),
                "sample_type": values.get("sample_type"),
            }
            top_level = {
                "region_name": values.get("region_name"),
                "sample_spec": sample_spec,
            }
            return top_level
        else:
            return values
