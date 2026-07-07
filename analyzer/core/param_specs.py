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


@define
class ParameterSpec:
    default_value: Any | None = None
    possible_values: Collection | None = None
    tags: set[str] = field(factory=set)
    param_type: type | None = None

    driven_by: dict[str, Callable[[str], str | None]] | None = None

    def getIndependentValues(self, full_spec: dict[str, "ParameterSpec"]) -> set:
        if not self.possible_values:
            return set()
        if not self.driven_by:
            return set(self.possible_values)

        driven_values = set()
        for driver_name, mapping_fn in self.driven_by.items():
            if driver_name not in full_spec:
                continue
            driver_spec = full_spec[driver_name]
            for driver_val in driver_spec.possible_values or []:
                result = mapping_fn(driver_val)
                if result is not None:
                    driven_values.add(result)

        return set(self.possible_values) - driven_values


def getTags(multi_spec, *tags):
    return {x: y for x, y in multi_spec.items() if any(t in y.tags for t in tags)}


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


ModuleParameterSpec = dict[str, ParameterSpec]
