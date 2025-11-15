from __future__ import annotations
import copy
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
    ret = [dict(zip(s, f)) for s,f in zip(outfiles, outsites)]
    return ret


class SampleType(str, enum.Enum):
    MC = "MC"
    Data = "Data"



@define
class Sample:
    sample_name: str
    n_events: int
    source: SourceDescription
    x_sec: float | None = None 

    @property
    def metdata(self):
        return dict(sample_name=self.sample_name, x_sex=self.x_sec)

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
        found = next(x for x in self.samples if x.sample_name == sample_name)
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
