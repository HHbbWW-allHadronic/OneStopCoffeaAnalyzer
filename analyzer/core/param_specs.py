from __future__ import annotations
from typing import Callable
from rich import print
from attrs import define, field
from analyzer.utils.structure_tools import deepMerge
from collections.abc import Collection
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

def getTags(multi_spec, *tags):
    return {
        x: y for x, y in multi_spec.items() if any(t in y.tags for t in tags)
    }

def getWithValues(multi_spec, values: dict[str, Any]):
    ret = {}
    for name, spec in multi_spec.items():
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
    

def toTuples(d):
    return {(x, y): v for x, s in d.items() for y, v in s.items()}


ModuleParameterSpec = dict[str,ParameterSpec]
