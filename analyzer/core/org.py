import inspect
import logging
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from graphlib import CycleError, TopologicalSorter

logger = logging.getLogger(__name__)


class AnalyzerGraphError(Exception):
    def __init__(self, message):
        super().__init__(message)


def iterableNotStr(t):
    return isinstance(t, Iterable) and not isinstance(t, str)


def toSet(x):
    if iterableNotStr(x):
        return set(x)
    else:
        return (x,)


def isMC(sample_info):
    return sample_info.sample_type == "MC"


class AnalyzerModule:
    def __init__(
        self,
        name,
        function,
        depends_on=None,
        categories="main",
        after=None,
        always=False,
        dataset_pred=None,
        documentation=None,
        processing_info=None,
    ):
        self.name = name
        self.function = function
        self.depends_on = toSet(depends_on) if depends_on else set()
        self.categories = toSet(categories) if categories else set()
        self.dataset_pred = dataset_pred or (lambda x: True)
        self.always = always
        self.documenation = documentation
        self.processing_info = processing_info or {}

    def __call__(self, events, analyzer):
        return self.function(events, analyzer)

    def __str__(self):
        return f"AnalyzerModule({self.name}, depends_on={self.depends_on}, categories={self.categories})"

    def __repr__(self):
        return str(self)


modules = {}
category_after = {
    "selection": ["preselection"],
    "post_selection": ["selection"],
    "apply_selection": ["selection"],
    "post_selection" : ["apply_selection"],
    "weights": ["post_selection"],
    "category": ["post_selection"],
    "finalize_weights": ["post_selection", "weights"],
    "main": ["post_selection", "weights", "category"],
    "final": ["main"],
}


def generateTopology(module_list, sample_info, include_defaults=True):
    logger.debug(f"Including defaults: {include_defaults}")
    if not include_defaults:
        logger.warn(
            f"Not including default modules. This may lead to unexpected behavior, use only if you are certain this is what you want to do!"
        )
    logger.info(f"Resolving modules")
    ts = tuple(reversed(tuple(TopologicalSorter(category_after).static_order())))
    resolved_category_after = {ts[i]: ts[i + 1 :] for i in range(len(ts))}
    logger.info(f"Category depencies have been resolved to\n{resolved_category_after}")

    mods = [x for x in module_list if x.dataset_pred(sample_info)]

    if include_defaults:
        all_mods = [x for x in modules.values() if x.always]
        logger.info(
            f"Adding {len(all_mods)} default modules:\n{[x.name for x in all_mods]}"
        )
        mods.extend([x for x in modules.values() if x.always])

    mods = list(set(mods))

    logger.info(f"Unfiltered modules are:\n{[x.name for x in mods]}")
    diff = [x.name for x in mods if not x.dataset_pred(sample_info)]
    mods = [x.name for x in mods if x.dataset_pred(sample_info)]
    logger.info(
        f"Dropped {len(diff)} modules  because of incompatible dataset predicate:\n{diff}"
    )
    logger.info(f"Filtered modules are:\n{mods}")

    cats = defaultdict(list)

    for x in [modules[n] for n in mods]:
        for c in x.categories:
            cats[c].append(x.name)

    graph = {}

    for name in mods:
        module = modules[name]
        graph[name] = module.depends_on
        for c in module.categories:
            for a in resolved_category_after.get(c, []):
                graph[name].update(set(cats[a]))

        for m in graph[name]:
            if m not in mods:
                raise AnalyzerGraphError(
                    f"Module {name} depends on {m}, but was this dependency was not supplied"
                )

    return graph


def namesToModules(module_list):
    return [modules[x] for x in module_list]


def sortModules(module_list, sample_info, include_defaults=True):
    graph = generateTopology(module_list, sample_info, include_defaults)
    try:
        ts = TopologicalSorter(graph)
        ret = tuple(ts.static_order())
    except CycleError as e:
        raise AnalyzerGraphError(
            f"Cyclic dependency detected in module specification:\n {' -> '.join(e.args[1])}\n"
            f"You may need to reorder the modules."
        )
    return namesToModules(ret)


def analyzerModule(name, **kwargs):
    def decorator(func):
        if name in modules:
            raise KeyError(f"A module already exists with the name {name}")

        modules[name] = AnalyzerModule(name, func, documentation=func.__doc__, **kwargs)
        return func

    return decorator
