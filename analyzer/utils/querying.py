from __future__ import annotations
from enum import Enum
from fnmatch import fnmatch
import re
from typing import Annotated
from collections.abc import Generator
from rich import print
import string


from pydantic import Field

from pydantic import BaseModel, RootModel, model_validator, TypeAdapter, field_validator


def modelIter(model: BaseModel) -> Generator[tuple[str, Any], None, None]:
    for field, value in model:
        if isinstance(value, BaseModel):
            for key, val in modelIter(value):
                yield f"{field}.{key}", val
        else:
            yield field, value


def lookup(obj, key):
    try:
        return getattr(obj, key)
    except AttributeError as e:
        try:
            return getattr(obj, "__getitem__")(key)
        except (KeyError, AttributeError):
            raise e


def deepLookup(obj, key):
    try:
        return lookup(obj, key)
    except AttributeError as e:
        parts = key.split(".", 1)
        if len(parts) == 1:
            raise e
        else:
            return deepLookup(lookup(obj, parts[0]), parts[1])
    return ret


class PatternMode(str, Enum):
    REGEX = "REGEX"
    GLOB = "GLOB"
    LITERAL = "LITERAL"
    ANY = "ANY"


class Pattern(BaseModel):
    mode: PatternMode = PatternMode.GLOB
    pattern: str | int | float

    def match(self, data, strict=True):
        if self.mode == PatternMode.ANY:
            return True
        elif self.mode == PatternMode.REGEX:
            return re.match(self.pattern, str(data))
        elif self.mode == PatternMode.GLOB:
            return fnmatch(str(data), self.pattern)
        else:
            return self.pattern == datastring

    def capture(self, data):
        if self.match(data):
            return data
        else:
            return None

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
            return {"mode": "LITERAL", "pattern": data}
        return data

    @staticmethod
    def Any():
        return Pattern(pattern="", mode=PatternMode.ANY)


class MultiPatternOp(str, Enum):
    AND = "AND"
    OR = "OR"


class MultiPatternExpression(BaseModel):
    op: MultiPatternOp
    exprs: list[PatternExpression]

    def match(self, data, strict=True):
        mapping = {MultiPatternOp.AND: all, MultiPatternOp.OR: any}
        return mapping[self.op](x.match(data,strict=strict) for x in self.exprs)

    def capture(self, data):
        captures = [x.capture(data) for x in self.exprs]
        mapping = {MultiPatternOp.AND: all, MultiPatternOp.OR: any}
        ok = mapping[self.op](x is not None for x in captures)
        if ok:
            return next(x for x in captures if x is not None)
        else:
            return None

    @model_validator(mode="before")
    @classmethod
    def handleList(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"op": MultiPatternOp.OR, "exprs": data}
        else:
            return data


class ComplexNestedPatternExpression(RootModel):
    root: dict[str, UnaryPatternExpression | MultiPatternExpression | Pattern]

    def match(self, data, strict=True):
        ret = True
        for k, pattern in self.root.items():
            try:
                found = deepLookup(data, k)
                ret = ret and pattern.match(found, strict=strict)
            except AttributeError:
                if strict:
                    return False
        return ret

    def fields(self):
        return list(self.root)

    def capture(self, data):
        ret = {}
        for k, pattern in self.root.items():
            try:
                found = deepLookup(data, k)
                captured = pattern.capture(found)
                ret[k] = captured
            except AttributeError:
                ret[k] = None
        return ret


SimpleNestedPatternExpression = ComplexNestedPatternExpression


# class SimpleNestedPatternExpression(RootModel):
#     root: dict[str, Pattern]
#
#     def match(self, data):
#         ret = True
#         for k, pattern in self.root.items():
#             try:
#                 found = deepLookup(data, k)
#                 ret = ret and pattern.match(found)
#             except AttributeError:
#                 return False
#         return ret
#
#     def capture(self, data):
#         ret = {}
#         for k, pattern in self.root.items():
#             try:
#                 found = deepLookup(data, k)
#                 captured = pattern.capture(found)
#                 ret[k] = captured
#             except AttributeError:
#                 ret[k] = None
#         return ret
#
#     def fields(self):
#         return list(self.root)


class UnaryPatternOp(str, Enum):
    NOT = "NOT"


class UnaryPatternExpression(BaseModel):
    op: UnaryPatternOp
    expr: PatternExpression

    def match(self, data):
        if self.op == UnaryPatternOp.NOT:
            return not self.expr.match(data)

    def capture(self, data):
        if self.match(data):
            return data
        else:
            return None


NestedPatternExpression = ComplexNestedPatternExpression


def model_x_discriminator(v: Any) -> str:
    if isinstance(v, int):
        return "int"
    if isinstance(v, (dict, BaseModel)):
        return "model"
    else:
        # return None if the discriminator value isn't found
        return None


PatternExpression = Annotated[
    Pattern | MultiPatternExpression | UnaryPatternExpression | NestedPatternExpression,
    Field()
]
pattern_expr_adapter = TypeAdapter(PatternExpression)

# PatternList = TypeAdapter(list[Pattern])
# QueryPattern = TypeAliasType("QueryPattern", Pattern | dict[str, "QueryPattern"])
# QueryPatternAdapter = TypeAdapter(QueryPattern)
