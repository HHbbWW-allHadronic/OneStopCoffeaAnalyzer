from __future__ import annotations
import re
import dataclasses
import enum
from analyzer.core.event_collection import SourceDescription
from rich.progress import track
import logging
from pathlib import Path
from analyzer.core.serialization import converter
from typing import Any
from attrs import define, field

import yaml
from yaml import CLoader as Loader
from analyzer.configuration import CONFIG


from .era import Era

logger = logging.getLogger(__name__)


def getDatasets(query, client):
    from coffea.dataset_tools import rucio_utils

    outlist, outtree = rucio_utils.query_dataset(
        query,
        client=client,
        tree=True,
        scope="cms",
    )
    return outlist


def getReplicas(dataset, client):
    from analyzer.utils.file_tools import extractCmsLocation
    from coffea.dataset_tools import rucio_utils

    (
        outfiles,
        outsites,
        sites_counts,
    ) = rucio_utils.get_dataset_files_replicas(
        dataset,
        allowlist_sites=[],
        blocklist_sites=["T3_CH_CERN_OpenData"],
        regex_sites=[],
        mode="full",  # full or first. "full"==all the available replicas
        client=client,
    )
    ret = {}
    for f, s in zip(outfiles, outsites):
        ret[extractCmsLocation(f[0])] = dict(zip(s, f))
    return ret


class SampleType(str, enum.Enum):
    MC = "MC"
    Data = "Data"


# @defgine
# class DatasetParams:
#     model_config = ConfigDict(use_enum_values=True)
#
#     name: str
#     sample_type: SampleType
#     title: str
#     era: str | dict
#     other_data: dict[str, Any] = Field(default_factory=dict)
#
#     _lumi: float | None = None
#
#     @property
#     def lumi(self) -> float:
#         if isinstance(self.era, str) and not self._lumi:
#             raise RuntimeError(f"Cannot compute lumi for dataset")
#         if self._lumi is not None:
#             return self._lumi
#         else:
#             return self.era.lumi
#
#     def populateEra(self, era_repo):
#         if isinstance(self.era, str):
#             self.era = era_repo[self.era]


# @pyd.dataclasses.dataclass(frozen=True)
# class SampleId:
#     dataset_name: str
#     sample_name: str
#
#     def __str__(self):
#         return self.serialize()
#
#     def __lt__(self, other):
#         return (self.dataset_name, self.sample_name) < (
#             other.dataset_name,
#             other.sample_name,
#         )
#
#     @pyd.model_serializer
#     def serialize(self) -> str:
#         return self.dataset_name + "___" + self.sample_name
#
#     @pyd.model_validator(mode="before")
#     @classmethod
#     def isStr(self, value):
#         if isinstance(value, str):
#             a, b, *rest = value.split("___")
#             return {"dataset_name": a, "sample_name": b}
#         elif len(value.args) == 1:
#             a, b, *rest = value.args[0].split("___")
#             return {"dataset_name": a, "sample_name": b}
#         else:
#             return value


# class SampleParams(BaseModel):
#     dataset: DatasetParams
#     name: str
#     n_events: int
#     x_sec: float | None = None
#     cms_dataset_regex: str | None = None
#     total_gen_weight: str | None = None
#     trigger_list: set[str] | None = None
#
#     @property
#     def sample_id(self):
#         return SampleId(dataset_name=self.dataset.name, sample_name=self.name)


@define
class Sample:
    sample_name: str
    source: SourceDescription
    x_sec: float | None = None 

    @property
    def metdata(self):
        return dict(sample_name=self.sample_name, x_sex=self.x_sec)

    # def useFilesFromReplicaCache(self):
    #     from analyzer.configuration import CONFIG
    #
    #     """Add files from the replica cache to the available files for this sample.
    #     """
    #
    #     replica_cache = Path(CONFIG.APPLICATION_DATA) / "replica_cache"
    #     look_for = replica_cache / f"{self.sample_id}.json"
    #     if not look_for.exists():
    #         return
    #     with open(look_for, "r") as f:
    #         replicas = json.load(f)
    #     t = list(it.chain.from_iterable(x.items() for x in replicas.values()))
    #     flat = dict(t)
    #     if len(flat) != len(self.files):
    #         raise RuntimeError(
    #             f"Possible missing files for {self._parent_dataset.name} - {self.name}."
    #             f"The number of files in the replica cache is {len(flat)}."
    #             f"The number of files in the dateset is {len(self.files)}."
    #         )
    #     for f in self.files:
    #         cms_loc = f.cmsLocation()
    #         for l, p in flat[cms_loc].items():
    #             f.setFile(l, p)
    #
    # def discoverAndCacheReplicas(self, force=False):
    #     """Use rucio to identify replicas for this sample, and store them for later use."""
    #
    #     from analyzer.configuration import CONFIG
    #     from coffea.dataset_tools import rucio_utils
    #
    #     if not self.cms_dataset_regex:
    #         raise RuntimeError(
    #             "Cannot call discoverReplicas on a sample with no CMS dataset"
    #         )
    #
    #     replica_cache = Path(CONFIG.APPLICATION_DATA) / "replica_cache"
    #     look_for = replica_cache / f"{self.sample_id}.json"
    #
    #     if look_for.exists() and not force:
    #         return
    #
    #     client = rucio_utils.get_rucio_client()
    #     datasets = getDatasets(self.cms_dataset_regex, client)
    #     replicas = {dataset: getReplicas(dataset, client) for dataset in datasets}
    #     look_for.parent.mkdir(exist_ok=True, parents=True)
    #     with open(look_for, "w") as f:
    #         json.dump(replicas, f, indent=2)
    #
    # def getFileSet(self, file_retrieval_kwargs):
    #     ret = {
    #         f: (
    #             d,
    #             {
    #                 "object_path": d.object_path,
    #                 "num_entries": None,
    #                 "uuid": None,
    #                 "steps": None,
    #             },
    #         )
    #         for f, d in self.fdict.items()
    #     }
    #     return FileSet(
    #         files=ret,
    #         step_size=None,
    #         form=None,
    #         file_retrieval_kwargs=file_retrieval_kwargs,
    #     )


@define
class Dataset:
    dataset_name: str
    title: str
    samples: list[Sample] 
    era: str 
    sample_type: SampleType
    other_data: dict[str, Any] = field(factory=dict)

    @property
    def metadata(self):
        return dict(
            dataset_name=self.dataset_name,
            title=self.title,
            era=self.era,
            other_data=self.other_data,
        )

    def getWithMeta(self, sample_name):
        current_meta = copy.copy(self.metadata)
        found = next(x for x in samples if x.sample_name == sample_name)
        current_meta.update(found.metadata)
        current_meta["sample_id"] = (
            self.metadata["dataset_name"] + "__" + found.metadata["sample_name"]
        )
        return current_meta, found

@define
class DatasetRepo:
    datasets: dict[str, Dataset] = field(factory=dict)
    metadata: dist[str,Any] = field(factory=dict)

    def getWithMeta(self, key):
        found = self.datasets[key]
        current_meta = copy.copy(self.metadata)
        current_meta.update(found.metadata)
        return current_meta, found

    def addFromFile(self, path):
        with open(path, "r") as fo:
            data = yaml.load(fo, Loader=Loader)
        data = converter.structure(data, list[Dataset])
        for d in data:
            if d.dataset_name in self.datasets:
                raise KeyError(f"A dataset with the name {d.name} already exists")
            self.datasets[d.dataset_name] = d

    def addFromDirectory(self, path):
        directory = Path(path)
        files = list(directory.rglob("*.yaml"))
        for f in files:
            self.addFromFile(f)
