from __future__ import annotations
import functools as ft
from typing_extensions import TypeAliasType
import logging
from enum import Enum
from fnmatch import fnmatch
import re

from pydantic import (
    BaseModel,
    RootModel,
    ConfigDict,
    field_validator,
    model_validator,
    TypeAdapter,
)


class PatternMode(str, Enum):
    REGEX = "REGEX"
    GLOB = "GLOB"
    NA = "NA"


class Pattern(BaseModel):
    mode: PatternMode = PatternMode.GLOB
    pattern: str | int | float

    def match(self, string, *args, **kwargs):
        if isinstance(self.pattern, str):
            if self.mode == PatternMode.REGEX:
                ret = re.match(self.pattern, string, *args, **kwargs)
            else:
                ret = fnmatch(string, self.pattern)
            return ret
        else:
            return self.pattern == string

    @model_validator(mode="before")
    @classmethod
    def convertString(cls, data):
        if isinstance(data, str):
            if data.startswith("re:"):
                return {"mode": "REGEX", "pattern": data.removeprefix("re:")}
            elif data.startswith("glob:"):
                return {"mode": "GLOB", "pattern": data.removeprefix("glob:")}
            else:
                return {"mode": "GLOB", "pattern": data}
        if isinstance(data, int | float):
            return {"mode": "NA", "pattern": data}
        else:
            return data


PatternList = TypeAdapter(list[Pattern])
QueryPattern = TypeAliasType("QueryPattern", Pattern | dict[str, "QueryPattern"])
QueryPatternAdapter = TypeAdapter(QueryPattern)


def getNested(d, s):
    if s in d:
        return d[s]
    parts = s.split(".", 1)
    try:
        ret = getNested(d[parts[0]], parts[1])
    except KeyError as e:
        raise KeyError(str(s))
    return ret


def _dictMatches(pattern, data):
    if isinstance(pattern, Pattern):
        if pattern.match(data):
            return data
        else:
            return None
    elif isinstance(pattern, dict):
        if not isinstance(data, dict):
            raise ValueError(f'Data "{data}" is not a dict')
        ret = {}
        for k, p in pattern.items():
            r = _dictMatches(p, getNested(data, k))
            if r is not None:
                ret[k] = r
            else:
                return None
        return ret
    else:
        raise ValueError(f"Pattern must be a QueryPattern")


def dictMatches(pattern, data):
    r = QueryPatternAdapter.validate_python(pattern)
    print(r)
    return _dictMatches(r, data)


def dictOverlap(d1, d2):
    if isinstance(d1, dict):
        ret = {}
        for k in d1:
            if k in d2:
                v = dictOverlap(getNested(d1, k), getNested(d2, k))
                if v:
                    ret[k] = v
        return ret or None

    else:
        if d1 == d2:
            return d1
        else:
            return None
    return None
