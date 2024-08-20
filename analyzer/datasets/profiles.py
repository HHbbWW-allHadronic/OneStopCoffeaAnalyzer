from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from analyzer.utils import deepMerge
from copy import deepcopy
from functools import reduce

from yaml import dump, load

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


@dataclass
class Profile:
    name: str
    lumi_json: str
    btag_scale_factors: str
    hlt: List[str]
    pileup: Dict[str, str]

    @staticmethod
    def fromDict(name, data):
        return Profile(
            name=name,
            lumi_json=data.get("lumi_json"),
            btag_scale_factors=data.get("btag_scale_factors"),
            hlt=data["HLT"],
            pileup=data["pileup"],
        )


@dataclass
class ProfileRepo:
    profiles: Dict[str, Profile] = field(default_factory=dict)

    @staticmethod
    def __resolve(alldata, k):
        data = deepcopy(alldata[k])
        if "inherit" in data:
            inherit = data["inherit"]
            if isinstance(inherit, str):
                inherit = [inherit]
            to_merge = [deepcopy(ProfileRepo.__resolve(alldata, x)) for x in inherit]
            data = reduce(deepMerge, [*to_merge, data])
        return data

    def loadFromDirectory(self, directory, force_separate=False):
        directory = Path(directory)
        files = list(directory.glob("*.yaml"))
        file_contents = {}
        temp_profiles = {}
        for f in files:
            with open(f, "r") as fo:
                data = load(fo, Loader=Loader)
                for profile, vals in data.items():
                    temp_profiles[profile] = vals

        for profile in temp_profiles:
            if temp_profiles[profile].get("partial", False):
                continue
            data = ProfileRepo.__resolve(temp_profiles, profile)
            self.profiles[profile] = Profile.fromDict(profile, data)


    def __getitem__(self, key):
        return self.profiles[key]
