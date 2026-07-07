import functools as ft
from attrs import define
from typing import Any
from rich import print
from cattrs.strategies import include_subclasses, configure_tagged_union
import abc
from analyzer.utils.querying import BasePattern
import copy
from analyzer.core.param_specs import ModuleParameterSpec, getTags
import logging
from typing import Callable
from collections import defaultdict

logger = logging.getLogger("analyzer.core")


def _buildDrivenMap(
    spec: ModuleParameterSpec,
) -> dict[str, list[tuple[str, Callable[[str], str | None]]]]:
    """driver_param_name -> [(driven_param_name, mapping_fn), ...]."""
    driven_map = defaultdict(list)
    for param_name, param_spec in spec.items():
        if not param_spec.driven_by:
            continue
        for driver_name, mapping_fn in param_spec.driven_by.items():
            driven_map[driver_name].append((param_name, mapping_fn))
    return driven_map


def buildCombos(spec, tag):
    ret = []
    tup = getTags(spec, tag)
    central = {k: v.default_value for k, v in tup.items()}
    driven_map = _buildDrivenMap(spec)

    for k, v in tup.items():
        independent = v.getIndependentValues(spec)
        for p in independent:
            if p == v.default_value:
                continue
            c = copy.deepcopy(central)
            c[k] = p
            if k in driven_map:
                for driven_param, mapping_fn in driven_map[k]:
                    correlated_value = mapping_fn(p)
                    if correlated_value is not None:
                        c[driven_param] = correlated_value
                        logger.debug(
                            f"Correlation: {k}={p} -> {driven_param}={correlated_value}"
                        )

            ret.append(["_".join([k, p]), c])
    return ret


class DEFAULT_RUN_BUILDER:
    pass


@define
class RunBuilder(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, spec: ModuleParameterSpec, metadata
    ) -> list[tuple[Any, dict]]: ...

    def __add__(self, other):
        return MultiRunBuilder([self, other])


@define
class MultiRunBuilder(RunBuilder):
    components: list[RunBuilder]

    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:
        used_names = set()
        ret = []
        for x in self.components:
            new= x(spec, metadata)
            new = [x for x in new if x[0] not in used_names]
            ret.extend(new)
            used_names |= set(x[0] for x in new)
        return ret


@define
class CompleteSysts(RunBuilder):
    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:
        weights = buildCombos(spec, "weight_variation")
        shapes = buildCombos(spec, "shape_variation")
        all_vars = [("central", {})] + weights + shapes
        return all_vars


@define
class LimitSysts(RunBuilder):
    systs: BasePattern

    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:
        weights = buildCombos(spec, "weight_variation")
        shapes = buildCombos(spec, "shape_variation")
        all_vars = weights + shapes
        all_vars = [x for x in all_vars if self.systs.match(x[0])]
        if not any(x[0] == "central" for x in all_vars):
            all_vars = [("central", {})] + all_vars
        return all_vars

@define
class LimitSystsBackground(RunBuilder):
    systs: BasePattern

    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:

        if "signal" in metadata["dataset_name"]:
            return [("central", {})]
        weights = buildCombos(spec, "weight_variation")
        shapes = buildCombos(spec, "shape_variation")
        all_vars = weights + shapes
        all_vars = [x for x in all_vars if self.systs.match(x[0])]

        if not any(x[0] == "central" for x in all_vars):
            all_vars = [("central", {})] + all_vars
        return all_vars


@define
class WeightsOnly(RunBuilder):
    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:
        weights = buildCombos(spec, "weight_variation")
        all_vars = [("central", {})] + weights
        return all_vars


@define
class SignalOnlySysts(RunBuilder):
    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:
        if "signal" in metadata["dataset_name"] or metadata.get("is_signal"):
            weights = buildCombos(spec, "weight_variation")
            shapes = buildCombos(spec, "shape_variation")
            all_vars = [("central", {})] + weights + shapes
            return all_vars
        else:
            return [("central", {})]


@define
class NoSystematics(RunBuilder):
    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:
        return [("central", {})]


@define
class UnscaledOnly(RunBuilder):
    def __call__(self, spec: ModuleParameterSpec, metadata) -> list[tuple[Any, dict]]:
        return [("UNSCALED", {})]


def configureConverter(conv):
    union_strategy = ft.partial(configure_tagged_union, tag_name="strategy_name")
    include_subclasses(RunBuilder, conv, union_strategy=union_strategy)
