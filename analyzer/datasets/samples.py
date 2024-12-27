import dataclasses
import enum
import itertools as it
import json
import logging
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Union

import pydantic as pyd
import yaml
from analyzer.configuration import CONFIG
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
    ConfigDict,
)
import analyzer.datasets.prepped as adp
import analyzer.datasets.files as adf


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


class DatasetParams(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str
    title: str
    era: Union[str, Era]
    sample_type: SampleType
    other_data: dict[str, Any] = Field(default_factory=dict)
    skimmed_from: Optional[str] = None

    _lumi: Optional[float] = None

    @property
    def lumi(self) -> float:
        if isinstance(self.era, str) and not self._lumi:
            raise RuntimeError(f"Cannot compute lumi for dataset")
        if self._lumi is not None:
            return self._lumi
        else:
            return self.era.lumi

    def populateEra(self, era_repo):
        if isinstance(self.era, str):
            self.era = era_repo[self.era]


@pyd.dataclasses.dataclass
class SampleParams:
    dataset: DatasetParams
    sample: dict[str, Any]


@pyd.dataclasses.dataclass(frozen=True)
class SampleId:
    dataset_name: str
    sample_name: str

    def __str__(self):
        return self.dataset_name + "__" + self.sample_name

    def __lt__(self, other):
        return (self.dataset_name, self.sample_name) < (
            other.dataset_name,
            other.sample_name,
        )

    @pyd.model_serializer
    def serialize(self) -> str:
        return self.dataset_name + "___" + self.sample_name

    @pyd.model_validator(mode="before")
    @classmethod
    def isStr(self, value):
        if isinstance(value, str):
            a, b, *rest = value.split("___")
            return {"dataset_name": a, "sample_name": b}
        else:
            return value


class Sample(BaseModel):
    """A single sample.
    Each sample has a single weight based on its cross section and number of events.
    """

    name: str
    n_events: int
    x_sec: Optional[float] = None  # Only needed if SampleType == MC
    files: List[adf.SampleFile] = Field(default_factory=list)
    cms_dataset_regex: Optional[str] = None
    total_gen_weight: Optional[str] = None
    _parent_dataset: Optional["Dataset"] = None

    @cached_property
    def fdict(self):
        return {f.cmsLocation(): f for f in self.files}

    @property
    def era(self):
        return self._parent_dataset.era

    @property
    def sample_type(self):
        return self._parent_dataset.sample_type

    @property
    def other_data(self):
        return self._parent_dataset.other_data

    @property
    def sample_id(self):
        return SampleId(self._parent_dataset.name, self.name)

    @property
    def params(self):
        return SampleParams(
            dataset=self._parent_dataset.params,
            sample=self.dict(exclude=["files"]),
        )

    def useFilesFromReplicaCache(self):
        from analyzer.configuration import CONFIG

        """Add files from the replica cache to the available files for this sample.
        """

        replica_cache = Path(CONFIG.APPLICATION_DATA) / "replica_cache"
        look_for = replica_cache / f"{self.sample_id}.json"
        if not look_for.exists():
            return
        with open(look_for, "r") as f:
            replicas = json.load(f)
        t = list(it.chain.from_iterable(x.items() for x in replicas.values()))
        flat = dict(t)
        if len(flat) != len(self.files):
            raise RuntimeError(f"Possible missing files for {self.name}")
        for f in self.files:
            cms_loc = f.cmsLocation()
            for l, p in flat[cms_loc].items():
                f.setFile(l, p)

    def discoverAndCacheReplicas(self, force=False):
        """Use rucio to identify replicas for this sample, and store them for later use."""

        from analyzer.configuration import CONFIG
        from coffea.dataset_tools import rucio_utils

        if not self.cms_dataset_regex:
            raise RuntimeError(
                "Cannot call discoverReplicas on a sample with no CMS dataset"
            )

        replica_cache = Path(CONFIG.APPLICATION_DATA) / "replica_cache"
        look_for = replica_cache / f"{self.sample_id}.json"

        if look_for.exists() and not force:
            return

        client = rucio_utils.get_rucio_client()
        datasets = getDatasets(self.cms_dataset_regex, client)
        replicas = {dataset: getReplicas(dataset, client) for dataset in datasets}
        look_for.parent.mkdir(exist_ok=True, parents=True)
        with open(look_for, "w") as f:
            json.dump(replicas, f, indent=2)

    def getFileSet(self, file_retrieval_kwargs):
        ret = {
            f: (
                d,
                {
                    "object_path": d.object_path,
                    "num_entries": None,
                    "uuid": None,
                    "steps": None,
                },
            )
            for f, d in self.fdict.items()
        }
        return adp.FileSet(
            files=ret,
            step_size=None,
            form=None,
            file_retrieval_kwargs=file_retrieval_kwargs,
        )


class Dataset(BaseModel):
    """A single physics dataset.
    It may be comprised of one or more samples.
    For example the QCDInclusive sample is comprised of several HT binned samples.
    """

    name: str
    title: str
    era: str
    sample_type: SampleType
    samples: List[Sample] = Field(default_factory=list)
    lumi: Optional[float] = None
    other_data: dict[str, Any] = Field(default_factory=dict)
    skimmed_from: Optional[str] = None

    @property
    def params(self):
        ds = DatasetParams(**self.model_dump())
        ds._lumi = self.lumi
        return ds

    def getSample(self, name):
        return next(x for x in self.samples if x.name == name)

    def __getitem__(self, item):
        return self.getSample(item)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    @model_validator(mode="before")
    @classmethod
    def ifSingleton(cls, values):
        if "samples" not in values:
            sample = {
                "name": values["name"],
                "n_events": values["n_events"],
                "x_sec": values.get("x_sec"),
                "files": values["files"],
                "cms_dataset_regex": values.get("cms_dataset_regex"),
            }
            top_level = {
                "name": values["name"],
                "title": values["title"],
                "era": values["era"],
                "lumi": values.get("lumi"),
                "sample_type": values["sample_type"],
                "other_data": values.get("other_data", {}),
                "skimmed_from": values.get("skimmed_from"),
                "samples": [sample],
            }
            return top_level
        else:
            return values

    @model_validator(mode="after")
    def refParent(self):
        for sample in self.samples:
            sample._parent_dataset = self
            if (
                self.skimmed_from
                and self.sample_type == SampleType.MC
                and not sample.total_gen_weight
            ):
                raise ValueError(
                    f"Dataset {self.name} is marked as a skim, but "
                    f'it\'s subsample {sample.name} does not have a "total_gen_weight". '
                    f"Skimmed samples must contain this field. "
                    f'You can find the gen weight by running over the parent dataset "{self.skimmed_from}", '
                    f'and consulting the "total_gen_weight" in the result.'
                )
        return self

    @field_serializer("sample_type")
    def serialize_type(self, sample_type, _info):
        return sample_type.name

    @field_validator("sample_type")
    @classmethod
    def validate_type(cls, v, _info):
        return SampleType[v]


@dataclasses.dataclass
class DatasetRepo:
    datasets: dict[str, Dataset] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key):
        return self.datasets[key]

    def __contains__(self, key):
        return key in self.datasets

    def __iter__(self):
        return iter(self.datasets)

    def __getitem__(self, key):
        if isinstance(key, SampleId):
            return self.datasets[key.dataset_name][key.sample_name]
        else:
            return self.datasets[key]

    def getSample(self, sample_id):
        dataset = self[sample_id.dataset_name]
        sample = dataset.getSample(sample_id.sample_name)
        return sample

    def load(self, directory, use_replicas=True):
        directory = Path(directory)
        files = list(directory.rglob("*.yaml"))
        file_contents = {}
        for f in files:
            with open(f, "r") as fo:
                data = yaml.safe_load(fo)
                for d in data:
                    s = Dataset(**d)
                    if s.name in self.datasets:
                        raise KeyError(
                            f"Dataset name '{s.name}' is already use. Please use a different name for this dataset."
                        )
                    self.datasets[s.name] = s

        if use_replicas:
            self.useReplicaCache()

    def buildReplicaCache(self, force=False):
        for dataset in self.datasets.values():
            logger.info(f"Building replicas for {dataset}")
            for sample in dataset.samples:
                logger.info(
                    f'Attempting to build replices for {sample} with regex "{sample.cms_dataset_regex}"'
                )
                if sample.cms_dataset_regex:
                    sample.discoverAndCacheReplicas(force=force)

    def useReplicaCache(self):
        for dataset in self.datasets.values():
            for sample in dataset.samples:
                if sample.cms_dataset_regex:
                    sample.useFilesFromReplicaCache()

    @staticmethod
    def getConfig():
        paths = CONFIG.DATASET_PATHS
        repo = DatasetRepo()
        for path in paths:
            repo.load(path)
        return repo
