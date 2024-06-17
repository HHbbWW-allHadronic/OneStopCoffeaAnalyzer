import itertools as it
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from yaml import dump, load

import rich
from analyzer.core.inputs import AnalyzerInput
from analyzer.datasets.styles import Style
from coffea.dataset_tools.preprocess import DatasetSpec
from rich.table import Table

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


class ForbiddenDataset(Exception):
    pass


@dataclass
class SampleFile:
    paths: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def fromDict(data):
        if isinstance(data, Dict):
            return SampleFile(data)
        else:
            return SampleFile({"eos": data})

    def getRootDir(self):
        return "Events"

    def getFile(self, prefer_location=None, require_location=None):
        if prefer_location and require_location:
            raise ValueError(f"Cannot have both a preferred and required location")
        if require_location:
            try:
                return self.paths[require_location]
            except KeyError:
                raise KeyError(
                    f"Sample file does not have a path registered for location '{require_location}'.\n"
                    f"known locations are: {self.paths}"
                )
        return self.paths.get(prefer_location, next(iter(self.paths.values())))


@dataclass(frozen=True)
class SampleSet:
    name: str
    title: str
    derived_from: Optional[Union[str, "SampleSet"]]
    sample_type: Optional[str]
    produced_on: Optional[str]
    lumi: Optional[float]
    x_sec: Optional[float]
    n_events: int
    files: List[SampleFile] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    style: Optional[Union[Style, str]] = None
    isdata: bool = False
    forbid: Optional[bool] = False
    mc_campaign: Optional[str] = None
    lumi_json: Optional[str] = None
    lumi_json: Optional[str] = None

    @staticmethod
    def fromDict(data):
        name = data["name"]
        derived_from = data.get("derived_from", None)
        produced_on = data.get("produced_on", None)
        lumi = data.get("lumi", None)
        x_sec = data.get("x_sec", None)
        n_events = data.get("n_events", None)
        tags = set(data.get("tags", []))
        title = data.get("title", name)
        isdata = data.get("isdata", False)
        forbid = data.get("forbid", None)
        mc_campaign = data.get("mc_campaign", None)
        lumi_json = data.get("lumi_json", None)
        sample_type = data.get("sample_type", None)
        if not (x_sec and n_events and lumi) and not (derived_from or isdata):
            raise Exception(
                f"Every sample must either have a complete weight description, or a derivation. While processing\n {name}"
            )
        if isdata:
            if mc_campaign:
                raise Exception(f"A data sample cannot have an MC campaign.")
            if not lumi_json and not derived_from:
                raise Exception(
                    f"Data sample {name} does not have an associated lumi json"
                )

        files = [SampleFile.fromDict(x) for x in data["files"]]

        style = data.get("style", {})
        if not isinstance(style, str):
            style = Style.fromDict(style)

        ss = SampleSet(
            name,
            title,
            derived_from,
            sample_type,
            produced_on,
            lumi,
            x_sec,
            n_events,
            files,
            tags,
            style,
            isdata,
            forbid,
            mc_campaign,
            lumi_json,
        )
        return ss

    def isForbidden(self):
        if self.forbid is None:
            if self.derived_from is None:
                return False
            else:
                return self.derived_from.isForbidden()
        else:
            return self.forbid

    def getLumi(self):
        if self.derived_from:
            return self.derived_from.lumi
        else:
            return self.lumi

    def isData(self):
        if self.derived_from:
            return self.derived_from.isData()
        else:
            return self.isdata

    def getLumiJson(self):
        if self.derived_from:
            return self.derived_from.getLumiJson()
        else:
            return self.lumi_json

    def getXSec(self):
        if self.derived_from:
            return self.derived_from.getXSec()
        else:
            return self.x_sec

    def toCoffeaDataset(self, prefer_location=None, require_location=None):
        if self.isForbidden():
            raise ForbiddenDataset(
                f"Attempting to access the files for forbidden dataset {self.name}"
            )

        return {
            self.name: {
                "files": {
                    f.getFile(prefer_location, require_location): f.getRootDir()
                    for f in self.files
                }
            }
        }

    def getWeight(self, target_lumi=None):
        if self.derived_from:
            w = self.derived_from.getWeight()
        else:
            if self.isData():
                w = 1
            else:
                w = self.lumi * self.x_sec / self.n_events
        if target_lumi:
            w = w * target_lumi / self.getLumi()
        return w

    def getAnalyzerInput(
        self, setname=None, prefer_location=None, require_location=None
    ):
        return AnalyzerInput(
            dataset_name=self.name,
            fill_name=setname or self.name,
            coffea_dataset=self.toCoffeaDataset(prefer_location, require_location),
            lumi_json=self.getLumiJson(),
        )

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

    def getAnalyzerInput(self, prefer_location=None, require_location=None):
        return [
            x.getAnalyzerInput(
                None if self.treat_separate else self.name,
                prefer_location,
                require_location,
            )
            for x in self.getSets()
        ]

    def totalEvents(self):
        return sum(s.totalEvents() for s in self.sets)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)
