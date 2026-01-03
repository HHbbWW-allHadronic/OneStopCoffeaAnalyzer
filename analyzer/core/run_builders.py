from collections import defaultdict
import functools as ft
from attrs import define
from typing import Any
from cattrs.strategies import include_subclasses, configure_tagged_union
from cattrs import structure, unstructure
import abc
import copy

from analyzer.core.param_specs import PipelineParameterSpec


def toTuples(d):
    return {(x, y): v for x, s in d.items() for y, v in s.items()}


def fromTuples(d):
    ret = defaultdict(dict)
    for (k1, k2), v in d.items():
        ret[k1][k2] = v
    return dict(ret)


def buildCombos(spec, tag):
    ret = []
    tup = toTuples(spec.getTags(tag))
    central = {k: v.default_value for k, v in tup.items()}
    for k, v in tup.items():
        for p in v.possible_values:
            if p == v.default_value:
                continue
            c = copy.deepcopy(central)
            c[k] = p
            ret.append(["_".join([*k, p]), c])

    ret = [(n, fromTuples(x)) for n, x in ret]

    return ret


class DEFAULT_RUN_BUILDER:
    pass


@define
class RunBuilder(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, spec: PipelineParameterSpec, metadata
    ) -> list[tuple[Any, dict]]: ...

    def __add__(
        self, *args):
        return MultiRunBuilder(*args)

@define
class MultiRunBuilder(RunBuilder):
    components: list[RunBuilder]

    def __call__(self, spec: PipelineParameterSpec, metadata) -> list[tuple[Any, dict]]:
        ret = []
        for x in self.components:
            ret.extend(x(spec))
        return ret



@define
class CompleteSysts(RunBuilder):
    def __call__(self, spec: PipelineParameterSpec, metadata) -> list[tuple[Any, dict]]:
        weights = buildCombos(spec, "weight_variation")
        shapes = buildCombos(spec, "shape_variation")
        all_vars = [("central", {})] + weights + shapes
        return all_vars


@define
class SignalOnlySysts(RunBuilder):
    def __call__(self, spec: PipelineParameterSpec, metadata) -> list[tuple[Any, dict]]:
        if "signal" in metadata["dataset_name"] or metadata["is_signal"]:
            weights = buildCombos(spec, "weight_variation")
            shapes = buildCombos(spec, "shape_variation")
            all_vars = [("central", {})] + weights + shapes
            return all_vars
        else:
            return [("central", {})]



@define
class NoSystematics(RunBuilder):
    def __call__(self, spec: PipelineParameterSpec, metadata) -> list[tuple[Any, dict]]:
        return [("central", {})]


def configureConverter(conv):
    union_strategy = ft.partial(configure_tagged_union, tag_name="strategy_name")
    include_subclasses(RunBuilder, conv, union_strategy=union_strategy)
