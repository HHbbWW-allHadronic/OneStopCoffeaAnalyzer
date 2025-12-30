from __future__ import annotations
from typing import TypeVar, Generic
from analyzer.core.datasets import SampleType
from typing import Callable, Literal
from collections import OrderedDict
import functools as ft
from cattrs.strategies import include_subclasses, configure_tagged_union
from cattrs import structure, unstructure
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

logger = logging.getLogger("analyzer.core")

ModuleParameterValues = dict[str, Any]
PipelineParameterValues = dict[str, ModuleParameterValues]


@define
class ParameterSpec:
    default_value: Any | None = None
    possible_values: Collection | None = None
    tags: set[str] = field(factory=set)
    param_type: type | None = None
    correlation_function: Callable | None = None
    correlated_values: Collection | None = None

    @property
    def free_values(self):
        return set(self.possible_values or []) - set(self.correlated_values or [])


@define
class ModuleParameterSpec:
    param_specs: dict[str, ParameterSpec] = field(factory=dict)

    def getTags(self, *tags):
        return {
            x: y for x, y in self.param_specs.items() if any(t in y.tags for t in tags)
        }

    def __getitem__(self, key):
        return self.param_specs[key]

    def getWithValues(self, values: dict[str, Any]):
        ret = {}
        for name, spec in self.param_specs.items():
            if name in values:
                v = values[name]
                if (spec.possible_values is None or v in spec.possible_values) and (
                    spec.param_type is None or isinstance(v, spec.param_type)
                ):
                    ret[name] = values[name]
                else:
                    raise RuntimeError(
                        f"Value {v} not in the list of possible values for parameter {name}. Allowed values are {spec.possible_values}"
                    )
            else:
                if spec.default_value is None:
                    raise RuntimeError(
                        f"Must provide a value for {spec} -- {name} with no default value"
                    )
                ret[name] = spec.default_value
        return ret


@define
class PipelineParameterSpec:
    node_specs: dict[str, ModuleParameterSpec]

    def __getitem__(self, key):
        return self.node_specs[key]

    def __setitem__(self, key, value):
        if key in self.node_specs:
            raise RuntimeError()
        self.node_specs[key] = value

    def getWithValues(
        self, values: dict[str, dict[str, Any]], *rest: dict[str, dict[str, Any]]
    ):
        values = deepMerge(values, *rest, max_depth=1)
        ret = {}
        for nid, spec in self.node_specs.items():
            if nid in values:
                ret[nid] = spec.getWithValues(values[nid])
            else:
                ret[nid] = spec.getWithValues({})
        return ret

    def getTags(self, *tag):
        tags = {x: y.getTags(*tag) for x, y in self.node_specs.items()}
        return tags


def toTuples(d):
    return {(x, y): v for x, s in d.items() for y, v in s.items()}
