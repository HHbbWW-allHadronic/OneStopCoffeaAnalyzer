import logging
from fnmatch import fnmatch
from typing import Optional, Union, Any
import functools as ft

from collections import ChainMap
from analyzer.datasets import SampleId, SampleType, DatasetParams
from pydantic import BaseModel, ConfigDict
import pydantic as pyd

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
            a, *b = value.split("___")
            return {"region_name": a, "sample_id": "___".join(b)}
        else:
            return value



class SectorParams(pyd.BaseModel):
    dataset_params: DatasetParams
    region_params: dict[str, Any]

    @ft.cached_property
    def _all_params(self):
        def notNone(m):
            return {x: y for x, y in m.items() if y is not None}

        ret = ChainMap(
            notNone(self.dataset_params),
            notNone(self.region_params),
        )
        return ret

    def items(self):
        return self._all_params.items()

    def keys(self):
        return self._all_params.keys()

    def values(self):
        return self._all_params.values()

    def __getitem__(self, key):
        if key == "sector_id":
            return self.sector_id
        return self._all_params[key]

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError
        return self[attr]

class SubSectorParams(pyd.BaseModel):
    sector_params: SectorParams
    sample_params: dict[str, Any]
    subsector_id: SubSectorId

    @ft.cached_property
    def _all_params(self):
        def notNone(m):
            return {x: y for x, y in m.items() if y is not None}

        ret = ChainMap(
            notNone(self.sector_params),
            notNone(self.sample_params),
        )
        return ret

    def items(self):
        return self._all_params.items()

    def keys(self):
        return self._all_params.keys()

    def values(self):
        return self._all_params.values()

    def __getitem__(self, key):
        if key == "sector_id":
            return self.subsector_id
        return self._all_params[key]

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError
        return self[attr]


class SampleSpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: Optional[Union[list[str], str]] = None
    era: Optional[Union[list[str], str]] = None
    sample_type: Optional[SampleType] = None

    def passes(
            self, dataset_params, return_specificity=False, **kwargs
    ):
        passes_names = not self.name or any(fnmatch(dataset_params.name, x) for x in self.name)

        passes_era = not self.era or any(fnmatch(dataset_params.era, x) for x in self.era)
        passes_sample_type = not self.sample_type or (dataset_params.sample_type == self.sample_type)
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
    sample_spec: Optional[SampleSpec] = None
    region_names: Optional[Union[list[str], str]] = None

    def passes(self, sector_params,  region_name=None, return_specificity=False):
        passes_sample = not self.sample_spec or self.sample_spec.passes(
            sector_params.dataset_params, return_specificity=return_specificity
        )
        passes_region = not self.region_names or any(
            fnmatch(sector_params.region_name, x) for x in self.region_names
        )
        passes = bool(passes_sample) and passes_region
        if return_specificity:
            exact_region = any(sector_params.region_name == x for x in self.region_names)
            return (
                80 * exact_region
                + 50 * (not exact_region and passes_region)
                + passes_sample
            )

        else:
            return passes
