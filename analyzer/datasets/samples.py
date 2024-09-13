import itertools as it
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from collections import OrderedDict
import operator as op
import random

import copy
from yaml import dump, load

import rich
import json
from analyzer.core.inputs import AnalyzerInput, SampleInfo
from analyzer.datasets.styles import Style
from analyzer.file_utils import stripPrefix
from analyzer.configuration import getConfiguration
from coffea.dataset_tools.preprocess import DatasetSpec
from rich.table import Table
from urllib.parse import urlparse, urlunparse
from analyzer.file_utils import extractCmsLocation
import re

from pydantic import BaseModel, Field, validator
import enum


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

    (
        outfiles,
        outsites,
        sites_counts,
    ) = rucio_utils.get_dataset_files_replicas(
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



class Sample(BaseObject):
    name: str,
    title: str,
    n_events: int,
    profile: Optional[Union[str, Profile]],
    derived_from: Optional[Union[str, "SampleSet"]],
    sample_type: Optional[str],
    produced_on: Optional[str],
    lumi: Optional[float],
    x_sec: Optional[float],
    files: List[SampleFile] = field(default_factory=list),
    style: Optional[Union[Style, str]] = None,
    cms_dataset_regex: Optional[str] = None,




        self.name = name
        self.title = title
        self.sample_type = sample_type
        self.n_events = n_events
        self.cms_dataset_regex = cms_dataset_regex
        self.derived_from = derived_from

        self.__profile = profile
        self.__lumi = lumi
        self.__x_sec = x_sec

        self.files = files
        self.style = style
        self.required_modules = required_modules

    @staticmethod
    def fromDict(data):
        name = data["name"]
        derived_from = data.get("derived_from", None)
        produced_on = data.get("produced_on", None)
        profile = data.get("profile", None)
        if profile:
            profile = str(profile)
        lumi = data.get("lumi", None)
        x_sec = data.get("x_sec", None)
        n_events = data.get("n_events", None)
        title = data.get("title", name)
        isdata = data.get("isdata", False)
        forbid = data.get("forbid", None)
        mc_campaign = data.get("mc_campaign", None)
        required_modules = data.get("required_modules", None)

        sample_type = (
            data.get("sample_type", None)
            or ("MC" if mc_campaign else None)
            or ("Data" if isdata else None)
        )

        cms_dataset = data.get("cms_dataset", None)
        if not (x_sec and n_events and lumi) and not (derived_from or isdata):
            raise Exception(
                f"Every sample must either have a complete weight description, or a derivation. While processing\n {name}"
            )
        if not derived_from and not profile:
            raise Exception(
                f"Every non-derived sample must have a profile. While processing\n {name}"
            )
        if isdata:
            if mc_campaign:
                raise Exception(f"A data sample cannot have an MC campaign.")

        files = [SampleFile.fromDict(x) for x in data["files"]]

        style = data.get("style", {})
        if not isinstance(style, str):
            style = Style.fromDict(style)

        ss = SampleSet(
            name=name,
            title=title,
            n_events=n_events,
            derived_from=derived_from,
            profile=profile,
            sample_type=sample_type,
            produced_on=produced_on,
            lumi=lumi,
           x_sec=x_sec,
            files=files,
            style=style,
            isdata=isdata,
            forbid=forbid,
            mc_campaign=mc_campaign,
            cms_dataset_regex=cms_dataset,
            required_modules=required_modules,
        )
        return ss

    def useFilesFromReplicaCache(self):
        config = getConfiguration()
        replica_cache = Path(config["APPLICATION_DATA"]) / "replica_cache"
        look_for = replica_cache / f"{self.name}.json"
        if not look_for.exists():
            return
        with open(look_for, "r") as f:
            replicas = json.load(f)
        t = list(it.chain.from_iterable(x.items() for x in replicas.values()))
        flat = dict(t)
        # if len(flat) != len(self.files):
        #    raise RuntimeError()
        for f in self.files:
            cms_loc = f.cmsLocation()
            for l, p in flat[cms_loc].items():
                f.setFile(l, p)

    def discoverAndCacheReplicas(self, force=False):
        from coffea.dataset_tools.dataset_query import DataDiscoveryCLI
        from coffea.dataset_tools import rucio_utils

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


    @property
    def lumi(self):
        if self.derived_from:
            return self.derived_from.lumi
        else:
            return self.__lumi

    @property
    def x_sec(self):
        if self.derived_from:
            return self.derived_from.x_sec
        else:
            return self.__x_sec

    @property
    def profile(self):
        if self.derived_from and self.__profile is None:
            return self.derived_from.profile
        else:
            return self.__profile

    @profile.setter
    def profile(self, val):
        self.__profile = val

    def weight(self, target_lumi=None):
        if self.derived_from:
            w = self.derived_from.weight()
        else:
            if self.sample_type:
                w = 1
            else:
                w = self.lumi * self.x_sec / self.n_events
        if target_lumi:
            w = w * target_lumi / self.lumi
        return w

    def getAnalyzerInput(self, setname=None, **kwargs):
        return AnalyzerInput(
            dataset_name=self.name,
            last_ancestor=self.getLastAncestor().name,
            fill_name=setname or self.name,
            files={f.cmsLocation(): f for f in self.files},
            profile=self.profile,
            required_modules=self.required_modules,
            sample_info=SampleInfo(self.name, self.sample_type, self.mc_campaign),
        )

    def getLastAncestor(self):
        if self.derived_from:
            return self.derived_from.getLastAncestor()
        else:
            return self

    def totalEvents(self):
        return self.n_events

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class SampleCollection:
    name: str
    title: str
    sets: List[SampleSet] = field(default_factory=list)
    treat_separate: bool = False
    style: Optional[Union[Style, str]] = None

    @staticmethod
    def fromDict(data, manager, force_separate=False):
        name = data["name"]
        sets = data["sets"]
        title = data.get("title", name)
        real_sets = [manager.sets[s] for s in sets]
        style = data.get("style", {})
        if not isinstance(style, str):
            style = Style.fromDict(style)

        sc = SampleCollection(
            name,
            title,
            real_sets,
            data.get("treat_separate", False or force_separate),
            style,
        )

        return sc

    def getSets(self):
        return self.sets

    def getAnalyzerInput(self, **kwargs):
        return [
            x.getAnalyzerInput(
                None if self.treat_separate else self.name,
                **kwargs,
            )
            for x in self.getSets()
        ]

    def totalEvents(self):
        return sum(s.totalEvents() for s in self.sets)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)
