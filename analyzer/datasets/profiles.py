from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from yaml import dump, load

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


@dataclass
class Profile:
    name: str
    btag_working_points: field(default_factory=dict)
    lumi_json: str
    btag_scale_factors: str
    hlt: List[str]

    @staticmethod
    def fromDict(data):
        return Profile(
            name=data["profile_name"],
            btag_working_points=data["btag_working_points"],
            lumi_json=data.get("lumi_json"),
            btag_scale_factors=data.get("btag_scale_factors"),
            hlt=data["HLT"],
        )


@dataclass
class ProfileRepo:
    profiles: Dict[str, Profile] = field(default_factory=dict)

    def loadFromDirectory(self, directory, force_separate=False):
        directory = Path(directory)
        files = list(directory.glob("*.yaml"))
        file_contents = {}
        for f in files:
            with open(f, "r") as fo:
                data = load(fo, Loader=Loader)
                self.profiles[data["profile_name"]] = Profile.fromDict(data)

    def __getitem__(self, key):
        return self.profiles[key]
