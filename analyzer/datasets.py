import itertools as it
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from yaml import dump, load

import rich
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
class AnalyzerInput:
    dataset_name: str
    fill_name: str
    coffea_dataset: DatasetSpec
    lumi_json: Optional[str] = None


@dataclass
class Style:
    color: Optional[str]
    alpha: Optional[float]

    @staticmethod
    def fromDict(data):
        color = data.get("color")
        op = data.get("alpha")
        return Style(color, op)

    def keys(self):
        return (
            field.name
            for field in fields(self)
            if getattr(self, field.name) is not None
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def toDict(self):
        return dict(
            (field.name, getattr(self, field.name))
            for field in fields(self)
            if getattr(self, field.name) is not None
        )


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
            return self.paths[require_location]
        return self.paths.get(prefer_location, next(iter(self.paths.values())))


@dataclass(frozen=True)
class SampleSet:
    name: str
    title: str
    derived_from: Optional[Union[str, "SampleSet"]]
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


@dataclass
class SampleManager:
    sets: Dict[str, SampleSet] = field(default_factory=dict)
    collections: Dict[str, SampleCollection] = field(default_factory=dict)

    def possibleInputs(self):
        return [*self.sets, *self.collections]

    def getSet(self, name):
        return self.sets[name]

    def getCollection(self, name):
        return self.collections[name]

    def __getitem__(self, key):
        return self.sets.get(key, None) or self.collections[key]

    def loadSamplesFromDirectory(self, directory, force_separate=False):
        directory = Path(directory)
        files = list(directory.glob("*.yaml"))
        file_contents = {}
        for f in files:
            with open(f, "r") as fo:
                data = load(fo, Loader=Loader)
                file_contents[f] = data
            for d in [x for x in data if x.get("type", "") == "set" or "files" in x]:
                s = SampleSet.fromDict(d)
                if s.name in self.sets:
                    raise KeyError(
                        f"Dataset name '{s.name}' is already use. Please use a different name for this dataset."
                    )
                self.sets[s.name] = s
        for name, val in self.sets.items():
            derived = val.derived_from
            if derived:
                self.sets[name] = replace(val, derived_from=self.sets[derived])

        for data in file_contents.values():
            for d in [
                x for x in data if x.get("type", "") == "collection" or "sets" in x
            ]:
                s = SampleCollection.fromDict(d, self, force_separate)
                if s.name in self.sets:
                    raise KeyError(
                        f"SampleCollection name '{s.name}' is already used by a set. Please use a different name for this dataset."
                    )
                if s.name in self.collections:
                    raise KeyError(
                        f"SampleCollection name '{s.name}' is already used by a collection. Please use a different name for this dataset."
                    )
                self.collections[s.name] = s

        for x in it.chain(self.sets.values(), self.collections.values()):
            if isinstance(x.style, str):
                x.style = self[x.style].style


def createSampleAndCollectionTable(manager, re_filter=None):
    table = Table(title="Samples And Collections")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Number Events")
    everything = list(
        it.chain(
            zip(manager.sets.values(), it.repeat("Set")),
            zip(manager.collections.values(), it.repeat("Colletions")),
        )
    )
    if re_filter:
        p = re.compile(re_filter)
        everything = [x for x in everything if p.search(x[0].name)]
    for s, t in everything:
        table.add_row(s.name, t, str(s.totalEvents()))
    return table


def createSetTable(manager, re_filter=None):
    table = Table(title="Samples Sets")
    table.add_column("Name")
    table.add_column("Number Events")
    table.add_column("Data/MC")
    table.add_column("X-Sec")
    table.add_column("Lumi")
    table.add_column("Number Files")
    table.add_column("Derived From")
    everything = list(manager.sets.values())
    if re_filter:
        p = re.compile(re_filter)
        everything = [x for x in everything if p.search(x.name)]
    for s in everything:
        xs = s.getXSec()
        lumi = s.getLumi()
        table.add_row(
            s.name,
            f"{str(s.totalEvents())}",
            "Data" if s.isData() else "MC",
            f"{xs:0.2g}" if xs else "N/A",
            f"{lumi:0.4g}" if lumi else "N/A",
            f"{len(s.files)}",
            f"{s.derived_from.name}" if s.derived_from else "N/A",
        )
    return table


def createCollectionTable(manager, re_filter=None):
    table = Table(title="Samples Collections")
    table.add_column("Name")
    table.add_column("Number Events")
    table.add_column("Number Sets")
    table.add_column("Treat Separate")
    everything = list(manager.collections.values())
    if re_filter:
        p = re.compile(re_filter)
        everything = [x for x in everything if p.search(x.name)]
    for s in everything:
        table.add_row(
            s.name, f"{str(s.totalEvents())}", f"{len(s.sets)}", f"{s.treat_separate}"
        )
    return table
