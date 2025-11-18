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


@define
class EraRepo:
    eras: dict[str, dict] = field(factory=dict)

    def __getitem__(self, key):
        return self.eras[key]

    def addFromFile(self, path):
        with open(path, "r") as fo:
            data = yaml.load(fo, Loader=Loader)
        for d in data:
            name = d["name"]
            if name in self.eras:
                raise KeyError(f"A era with the name {name} already exists")
            self.eras[name] = d

    def addFromDirectory(self, path):
        directory = Path(path)
        files = list(directory.rglob("*.yaml"))
        for f in files:
            self.addFromFile(f)
