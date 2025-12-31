from __future__ import annotations
from typing import TypeVar, Generic, DefaultDict
from collections import defaultdict
from analyzer.core.datasets import SampleType
from typing import Callable, Literal
from analyzer.core.param_specs import (
    ParameterSpec,
    ModuleParameterSpec,
    PipelineParameterSpec,
    ModuleParameterValues,
)
from collections import OrderedDict
import functools as ft
from cattrs.strategies import (
    include_subclasses,
    configure_tagged_union,
    configure_union_passthrough,
)
from cattrs import structure, unstructure
from analyzer.core.run_builders import RunBuilder, DEFAULT_RUN_BUILDER
from rich import print
from attrs import define, field, make_class
from attrs import define, field
from analyzer.core.results import ResultBase
from analyzer.utils.structure_tools import freeze, mergeUpdate, deepMerge, SimpleCache
from collections.abc import Collection
from analyzer.core.columns import TrackedColumns, Column, ColumnCollection
import copy
import contextlib
import abc
from typing import Any
import logging
from enum import Enum
from typing import Any
from fnmatch import fnmatch
import re
import abc
from typing import Annotated
from collections.abc import Generator
from rich import print
import string
from attrs import field, define


def lookup(obj, key):
    try:
        return getattr(obj, key)
    except AttributeError as e:
        try:
            return getattr(obj, "__getitem__")(key)
        except (KeyError, AttributeError):
            raise e


def deepLookup(obj, key):
    current = obj
    for k in key:
        current = lookup(current, k)
    return current


class PatternMode(str, Enum):
    REGEX = "REGEX"
    GLOB = "GLOB"
    LITERAL = "LITERAL"
    ANY = "ANY"


NO_MATCH = object()


# @define
# class BasePattern(abc.ABC):
#     @abc.abstractmethod
#     def match(self, data, strict=True) -> bool: ...
#
#     @abc.abstractmethod
#     def capture(self, data) -> Any: ...


@define
class Pattern:
    pattern: str | int | float
    mode: PatternMode = PatternMode.GLOB

    def match(self, data, strict=True):
        if self.mode == PatternMode.ANY:
            return True
        elif self.mode == PatternMode.REGEX:
            return re.match(self.pattern, str(data))
        elif self.mode == PatternMode.GLOB:
            return fnmatch(str(data), self.pattern)
        else:
            return self.pattern == data

    def capture(self, data):
        if self.match(data):
            return data
        else:
            return NO_MATCH

    @classmethod
    def _structure(cls, data, conv):
        if isinstance(data, str):
            if data.startswith("re:"):
                data = {"mode": "REGEX", "pattern": data.removeprefix("re:")}
            elif data.startswith("glob:"):
                data = {"mode": "GLOB", "pattern": data.removeprefix("glob:")}
            else:
                data = {"mode": "GLOB", "pattern": data}
        if isinstance(data, int | float):
            data = {"mode": "LITERAL", "pattern": data}
        return cls(mode=data["mode"], pattern=data["pattern"])

    @staticmethod
    def Any():
        return Pattern(pattern="", mode=PatternMode.ANY)


@define
class PatternAnd:
    and_exprs: list[BasePattern]

    def match(self, data, strict=True):
        return all(x.match(data, strict=strict) for x in self.and_exprs)

    def capture(self, data):
        captures = [x.capture(data) for x in self.and_exprs]
        ok = all(x is not NO_MATCH for x in captures)
        if not ok:
            return NO_MATCH
        else:
            return captures


@define
class PatternOr:
    or_exprs: list[BasePattern]

    def match(self, data, strict=True):
        return any(x.match(data, strict=strict) for x in self.or_exprs)

    def capture(self, data):
        captures = [x.capture(data) for x in self.or_exprs]
        ok = any(x is not NO_MATCH for x in captures)
        if not ok:
            return NO_MATCH
        else:
            return next(x for x in captures if x is not NO_MATCH)


@define
class DeepPattern:
    key: tuple[str, ...]
    pattern: BasePattern

    def match(self, data, strict=True):
        item = deepLookup(data, self.key)
        return self.pattern.match(item, strict=strict)

    def capture(self, data):
        item = deepLookup(data, self.key)
        return {self.key: self.pattern.capture(item)}


BasePattern = Pattern | PatternOr | PatternAnd | DeepPattern


def configureConverter(conv):
    # union_strategy = ft.partial(configure_tagged_union)
    configure_union_passthrough(str | int | float, conv)
    base_hook = conv.get_structure_hook(BasePattern)
    pattern_hook = conv.get_structure_hook(Pattern)
    # include_subclasses(
    #     BasePattern, conv, subclasses=[DeepPattern, Pattern, PatternOr, PatternAnd]
    # )

    @conv.register_structure_hook
    def _(value, t) -> BasePattern:
        if isinstance(value, str | int | float):
            return pattern_hook(value, Pattern)
        else:
            return base_hook(value, t)


def gatherByCapture(pattern, items, key=lambda x: x.metadata):
    ret = {}
    for i in items:
        vals = pattern.capture(key(i))
        k = hash(freeze(vals))
        if k in ret:
            ret[k][1].append(i)
        else:
            ret[k] = [vals, [i]]
    return list(tuple(x) for x in ret.values())


@define
class MetaCaptureResult:
    groups: dict[str, Any]

@define
class MetaCaptureDesc:
    group_by: dict[str, BasePattern | MetaCaptureDesc]
    select: BasePattern | None = None

    def apply(self, items):
        if self.select:
            items = [x for x in items if self.select.match(x)]
        ret = {}
        for x, y in self.group_by.items():
            if isinstance(y, BasePattern):
                ret[x] = gatherByCapture(y, items)
            else:
                ret[x] = y.apply(items)
        return ret


def getCommonMeta(items):
    i = iter(items)
    ret = copy.deepcopy(dict(next(i)))
    for item in i:
        for k in item:
            if not (k in ret and ret[k] == item[k]):
                del ret[k]
    return ret
