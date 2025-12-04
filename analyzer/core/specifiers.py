import logging
import pydantic as pyd
from analyzer.datasets import DatasetParams, SampleId, SampleParams

logger = logging.getLogger("analyzer")


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


@pyd.dataclasses.dataclass(frozen=True)
class ShapeVariationId:
    column_name: str
    shape_variation: str


class SectorParams(pyd.BaseModel):
    dataset: DatasetParams
    region_name: str

    def simpleName(self):
        return f"({region_name}, {dataset.name})"
        


class SubSectorParams(pyd.BaseModel):
    sample: SampleParams
    region_name: str

    @property
    def dataset(self):
        return self.sample.dataset



# class SectorSpec(BaseModel):
#     sample_spec: SampleSpec | None = None
#     region_name: list[Pattern] | Pattern | None = None
# 
#     @field_validator("region_name")
#     @classmethod
#     def makeList(cls, val, info):
#         if val is None:
#             return val
#         if not isinstance(val, list):
#             return [val]
#         return val
# 
#     def passes(self, sector_params, return_specificity=False):
#         passes_sample = not self.sample_spec or self.sample_spec.passes(
#             sector_params.dataset, return_specificity=return_specificity
#         )
#         passes_region = not self.region_name or any(
#             x.match(sector_params.region_name) for x in self.region_name
#         )
#         passes = passes_sample and passes_region
#         if return_specificity:
#             exact_region = any(sector_params.region_name == x for x in self.region_name)
#             return (
#                 80 * exact_region
#                 + 50 * (not exact_region and passes_region)
#                 + passes_sample
#             )
# 
#         else:
#             return passes
# 
#     @model_validator(mode="before")
#     @classmethod
#     def ifSingleton(cls, values):
#         if "sample_spec" not in values:
#             sample_spec = {
#                 "name": values.get("sample_name"),
#                 "era": values.get("era"),
#                 "sample_type": values.get("sample_type"),
#             }
#             top_level = {
#                 "region_name": values.get("region_name"),
#                 "sample_spec": sample_spec,
#             }
#             return top_level
#         else:
#             return values
