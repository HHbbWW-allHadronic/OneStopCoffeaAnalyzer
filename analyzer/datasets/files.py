import itertools as it
import operator as op
import random
import re
from collections import OrderedDict
from collections.abc import Mapping
from urllib.parse import urlparse

from analyzer.utils.file_tools import extractCmsLocation
from pydantic import BaseModel, Field, model_validator


class SampleFile(BaseModel):
    paths: OrderedDict[str, str] = Field(default_factory=OrderedDict)
    object_path: str = "Events"

    @model_validator(mode="before")
    def ifFlat(cls, values):
        if isinstance(values, str):
            return {"paths": {"eos": values}, "object_path": "Events"}
        if isinstance(values, Mapping):
            if "object_path" in values:
                return values
            else:
                return {"paths": values, "object_path": "Events"}

    def __post_init__(self):
        self.cmsLocation()

    def setFile(self, location, url):
        if self.cmsLocation() != extractCmsLocation(url):
            raise ValueError(
                f'Url \'{url}\' does not have the same correct cms-location "{self.cmsLocation()}" != "{extractCmsLocation(url)}"'
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
