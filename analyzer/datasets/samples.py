import copy
import enum
import itertools as it
import json
import operator as op
import random
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse, urlunparse

import rich
from analyzer.configuration import getConfiguration
from analyzer.core.inputs import AnalyzerInput, SampleInfo
from analyzer.datasets.styles import Style
from analyzer.file_utils import extractCmsLocation, stripPrefix
from coffea.dataset_tools.preprocess import DatasetSpec
from pydantic import BaseModel, Field, validator
from rich.table import Table
from yaml import dump, load

from .profiles import Profile

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


class ForbiddenDataset(Exception):
    pass


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
    from coffea.dataset_tools import rucio_utils

    (outfiles, outsites, sites_counts,) = rucio_utils.get_dataset_files_replicas(
        dataset,
        allowlist_sites=[],
        blocklist_sites=[],
        regex_sites=[],
        mode="full",  # full or first. "full"==all the available replicas
        client=client,
    )
    ret = {}
    for f, s in zip(outfiles, outsites):
        ret[extractCmsLocation(f[0])] = dict(zip(s, f))
    return ret


class SampleFile(BaseModel):
    paths: OrderedDict[str, str] = field(default_factory=OrderedDict)
    object_path: str = "Events"
    # steps: Optional[List[List[int]]] = None
    # number_events: Optional[int] = None

    def __post_init__(self):
        self.cmsLocation()

    def setFile(self, location, url):
        if self.cmsLocation() != extractCmsLocation(url):
            raise ValueError(
                f"Url '{url}' does not have the same correct cms-location {self.cmsLocation()}"
            )
        self.paths[location] = url

    def getRootDir(self):
        return self.object_path

    def getFile(
        self,
        require_location=None,
        location_priority_regex=None,
        require_protocol=None,
        **kwargs,
    ):
        if location_priority_regex and require_location:
            location_priority_regex = None
            # raise ValueError(f"Cannot have both a preferred and required location")
        if require_protocol:
            paths = {
                k: v
                for k, v in self.paths.items()
                if urlparse(v)[0] == require_protocol
            }
        else:
            paths = self.paths

        if require_location is not None:
            try:
                return paths[require_location]
            except KeyError:
                raise KeyError(
                    f"Sample file does not have a path registered for location '{require_location}'.\n"
                    f"known locations are: {self.paths}"
                )

        def rank(val):
            x = list(
                (i for i, x in enumerate(location_priority_regex) if re.match(x, val)),
            )
            return next(iter(x), len(location_priority_regex))

        if location_priority_regex:
            sites_ranked = [(s, rank(s)) for s in paths.keys()]
            sites_ranked = list(sorted(sites_ranked, key=op.itemgetter(1)))
            groups = [
                (x, [z[0] for z in y])
                for x, y in it.groupby(sites_ranked, key=op.itemgetter(1))
            ]
            good = [
                x[1] for x in sorted((x for x in groups if x), key=op.itemgetter(0))
            ]
            s = next(iter(good), [])
        else:
            s = list(paths.keys())

        return paths[random.choice(s)]

    def getNarrowed(self, *args, **kwargs):
        f = self.getFile(*args, **kwargs)
        return SampleFile(
            {x: y for x, y in self.paths.items() if y == f},
            self.object_path,
            self.steps,
            self.number_events,
        )

    def cmsLocation(self):
        s = set(extractCmsLocation(x) for x in self.paths.values())
        if len(s) != 1:
            raise RuntimeError(
                f"File has more than 1 associated CMS location. This will cause problems since the file's identity can't be uniquely determined.\n{s}"
            )
        return next(iter(s))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.cmsLocation() == other
        elif isinstance(other, Sample):
            return self.cmsLocation() == other.cmsLocation()
        else:
            raise NotImplementedError()

    def __hash__(self):
        return hash(self.cmsLocation())


class SampleType(str, enum.Enum):
    MC = "MC"
    Data = "Data"


class SampleDescription(BaseObject):
    """A single sample.
    Each sample has a single weight based on its cross section and number of events.
    """

    parent_dataset: "Dataset"
    name: str
    n_events: int
    x_sec: Optional[float] = None # Only needed if SampleType == MC
    files: List[SampleFile] = field(default_factory=list)
    cms_dataset_regex: Optional[str] = None

    def useFilesFromReplicaCache(self):
        """Add files from the replica cache to the available files for this sample.
        """

        config = getConfiguration()
        replica_cache = Path(config["APPLICATION_DATA"]) / "replica_cache"
        look_for = replica_cache / f"{self.name}.json"
        if not look_for.exists():
            return
        with open(look_for, "r") as f:
            replicas = json.load(f)
        t = list(it.chain.from_iterable(x.items() for x in replicas.values()))
        flat = dict(t)
        if len(flat) != len(self.files):
            raise RuntimeError()
        for f in self.files:
            cms_loc = f.cmsLocation()
            for l, p in flat[cms_loc].items():
                f.setFile(l, p)

    def discoverAndCacheReplicas(self, force=False):
        """Use rucio to identify replicas for this sample, and store them for later use. 
        """

        from coffea.dataset_tools import rucio_utils
        from coffea.dataset_tools.dataset_query import DataDiscoveryCLI

        if not self.cms_dataset_regex:
            raise RuntimeError(
                "Cannot call discoverReplicas on a sample with no CMS dataset"
            )

        config = getConfiguration()
        replica_cache = Path(config["APPLICATION_DATA"]) / "replica_cache"
        look_for = replica_cache / f"{self.name}.json"

        if look_for.exists() and not force:
            return

        client = rucio_utils.get_rucio_client()
        datasets = getDatasets(self.cms_dataset_regex, client)
        replicas = {dataset: getReplicas(dataset, client) for dataset in datasets}
        look_for.parent.mkdir(exist_ok=True, parents=True)
        with open(look_for, "w") as f:
            json.dump(replicas, f, indent=2)




class Dataset(BaseModel):
    """A single physics dataset.
    It may be comprised of one or more samples.
    For example the QCDInclusive sample is comprised of several HT binned samples.
    """
    name: str
    title: str
    era: str
    sample_type: SampleType
    sets: List[Sample] = field(default_factory=list)
    lumi: Optional[float] = None  
    other_data: dict[str, Any] = field(default_factory=list)
    treat_separate: bool = False

