from __future__ import annotations



import re
from enum import Enum
from fnmatch import fnmatch
from typing import Any

from analyzer.utils.structure_tools import (
    freeze,
    ItemWithMeta,
)
from attrs import define
from cattrs.strategies import (
    configure_union_passthrough,
)


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
            try:
                ret = fnmatch(data, self.pattern)
            except TypeError:
                return False
            return ret
        else:
            return self.pattern == data

    def capture(self, data):
        if self.match(data):
            return data
        else:
            return NO_MATCH

    @classmethod
    def _structure(cls, data: str | int | float, conv):
        if isinstance(data, str):
            if data.startswith("re:"):
                data = {"mode": "REGEX", "pattern": data.removeprefix("re:")}
            elif data.startswith("glob:"):
                data = {"mode": "GLOB", "pattern": data.removeprefix("glob:")}
            else:
                data = {"mode": "GLOB", "pattern": data}
        if isinstance(data, int | float):
            data = {"mode": "LITERAL", "pattern": data}
        return cls(mode=PatternMode[data["mode"]], pattern=data["pattern"])

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
class PatternNot:
    not_expr: BasePattern

    def match(self, data, strict=True):
        return not (self.not_expr.match(data, strict=strict))

    def capture(self, data):
        matched = self.not_expr.match(data)
        if matched:
            return NO_MATCH
        else:
            return self.not_expr.capture(data)


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
        capture = self.pattern.capture(item)
        if capture is NO_MATCH:
            return capture
        return {self.key: capture}


BasePattern = Pattern | PatternOr | PatternAnd | DeepPattern | PatternNot


def configureConverter(conv):
    # union_strategy = ft.partial(configure_tagged_union)
    configure_union_passthrough(str | int | float, conv)
    base_hook = conv.get_structure_hook(BasePattern)
    pattern_hook = conv.get_structure_hook(Pattern)
    conv.get_structure_hook(DeepPattern)

    base_and_hook = conv.get_structure_hook(PatternAnd)

    @conv.register_structure_hook
    def andPatternHook(value: list, t) -> PatternAnd:
        return base_and_hook({"and_exprs": value}, t)

    and_hook = conv.get_structure_hook(PatternAnd)

    @conv.register_structure_hook
    def _(value, t) -> BasePattern:
        if isinstance(value, str | int | float):
            if isinstance(value, str) and len(value) > 0 and value[0] == "!":
                return PatternNot(not_expr=pattern_hook(value[1:], Pattern))
            else:
                return pattern_hook(value, Pattern)
        if isinstance(value, list):
            return and_hook(value, PatternAnd)
        return base_hook(value, t)

    base_hook2 = conv.get_structure_hook(BasePattern)

    @conv.register_structure_hook
    def _(value, t) -> BasePattern:
        try:
            return base_hook2(value, t)
        except Exception:
            [{k: v} for k, v in value.items()]
            return base_hook2(
                [{"key": k.split("."), "pattern": v} for k, v in value.items()], t
            )

    base_hook3 = conv.get_structure_hook(BasePattern)

    @conv.register_structure_hook
    def _(value, t) -> BasePattern | None:
        if value is None:
            return None
        return base_hook3(value, t)


@define
class CaptureSet:
    capture: Any
    items: list[ItemWithMeta]


def gatherByCapture(pattern, items, key=lambda x: x.metadata):
    ret = {}
    for i in items:
        vals = pattern.capture(key(i))
        k = hash(freeze(vals))
        if k in ret:
            ret[k][1].append(i)
        else:
            ret[k] = [vals, [i]]
    return list(CaptureSet(*x) for x in ret.values())
