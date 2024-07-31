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
from analyzer.core.inputs import AnalyzerInput
from .samples import SampleSet, SampleCollection
from .profiles import ProfileRepo

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


@dataclass
class SampleManager:
    weights = []
    weights_normalized = []
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

    def linkProfiles(self, profile_repo):
        for sample in self.sets:
            if self.sets[sample].profile:
                self.sets[sample].profile = profile_repo[self.sets[sample].profile]

    def loadSamplesFromDirectory(
        self, directory, profile_repo, force_separate=False, use_replicas=True
    ):
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

        self.linkProfiles(profile_repo)

        if use_replicas:
            self.useReplicaCache()

    def buildReplicaCache(self, force=False):
        for sample in self.sets.values():
            if sample.cms_dataset_regex:
                sample.discoverAndCacheReplicas(force=force)

    def useReplicaCache(self):
        for sample in self.sets.values():
            if sample.cms_dataset_regex:
                sample.useFilesFromReplicaCache()


