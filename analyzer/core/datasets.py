from __future__ import annotations
import copy
from rich import print
import re
import dataclasses
from typing import ClassVar
import enum
from analyzer.core.event_collection import SourceDescription
from rich.progress import track
import logging
from pathlib import Path
from analyzer.core.serialization import converter
from analyzer.core.caching import cache
from typing import Any
from attrs import define, field, fields

import yaml
from yaml import CLoader as Loader
from analyzer.configuration import CONFIG


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
    ret = [dict(zip(s, f)) for s, f in zip(outfiles, outsites)]
    return ret


class SampleType(str, enum.Enum):
    MC = "MC"
    Data = "Data"


@define
class Sample:
    name: str
    n_events: int
    source: SourceDescription
    x_sec: float | None = None

    @property
    def metadata(self):
        return dict(sample_name=self.name, x_sec=self.x_sec, n_events=self.n_events)


def configureConverter(conv):
    base_sample_hook = conv.get_structure_hook(Sample)

    @conv.register_structure_hook
    def sampleHook(value, type) -> Sample:
        if not "source" in value:
            f = set(x.name for x in fields(Sample))
            # source = conv.structure(value, SourceDescription)
            source = {}
            for k in list(value.keys()):
                if k not in f:
                    source[k] = value.pop(k)
            value["source"] = source
        ret= base_sample_hook(value)
        return ret

    base_dataset_hook = conv.get_structure_hook(Dataset)
    @conv.register_structure_hook
    def datasetHook(value, type) -> Dataset:
        if not "samples" in value:
            f = set(x.name for x in fields(Dataset))
            sample = {"name" : value["name"]}
            for k in list(value.keys()):
                if k not in f:
                    sample[k] = value.pop(k)
            value["samples"] = [sample]
        ret= base_dataset_hook(value)
        return ret


@define
class Dataset:
    name: str
    title: str
    samples: list[Sample]
    era: str
    sample_type: SampleType
    other_data: dict[str, Any] = field(factory=dict)

    @property
    def metadata(self):
        return dict(
            dataset_name=self.name,
            dataset_title=self.title,
            era=self.era,
            other_data=self.other_data,
        )

    @property
    def __iter__(self):
        return iter(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sample_name):
        found = next(x for x in self.samples if x.name == sample_name)
        return found


@cache.memoize(tag="dataset")
def getDatasetFromPathMTime(path, mtime):
    with open(path, "r") as fo:
        data = yaml.load(fo, Loader=Loader)
    data = converter.structure(data, list[Dataset])
    return data


@define
class DatasetRepo:
    datasets: dict[str, Dataset] = field(factory=dict)
    metadata: dict[str, Any] = field(factory=dict)

    def __getitem__(self, key):
        return self.datasets[key]

    def __iter__(self):
        return iter(self.datasets)

    def addFromFile(self, path):
        path = Path(path)
        data = getDatasetFromPathMTime(path, path.stat().st_mtime)
        for d in data:
            if d.name in self.datasets:
                raise KeyError(f"A dataset with the name {d.name} already exists")
            self.datasets[d.name] = d

    def addFromDirectory(self, path):
        directory = Path(path)
        files = list(directory.rglob("*.yaml"))
        for f in files:
            self.addFromFile(f)
