from graphlib import CycleError, TopologicalSorter
from collections import defaultdict
import logging
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


class AnalyzerModule:
    def __init__(
        self,
        name,
        function,
        depends_on=None,
        categories="main",
        after=None,
        always=False,
    ):
        self.name = name
        self.function = function
        self.depends_on = toSet(depends_on) if depends_on else set()
        self.categories = toSet(categories) if categories else set()
        self.always = always

    def __call__(self, events, analyzer):
        return self.function(events, analyzer)

    def __str__(self):
        return f"AnalyzerModule({self.name}, depends_on={self.depends_on}, catetories={self.categories})"

    def __repr__(self):
        return str(self)


modules = {}
category_after = {
    "post_selection": ["selection"],

    "weights": ["post_selection"],
    "category": ["post_selection"],
    "main": ["post_selection", "weights", "category"],
}


def generateTopology(module_list):
    mods = [x.name for x in module_list]
    mods.extend([x.name for x in modules.values() if x.always])

    cats = defaultdict(list)

    for x in [modules[n] for n in mods]:
        for c in x.categories:
            cats[c].append(x.name)

    graph = {}

    for name in mods:
        module = modules[name]
        graph[name] = module.depends_on
        for c in module.categories:
            for a in category_after.get(c, []):
                graph[name].update(set(cats[a]))

        for m in graph[name]:
            if m not in mods:
                raise AnalyzerGraphError(
                    f"Module {name} depends on {m}, but was this dependency was not supplied"
                )

    return graph


def namesToModules(module_list):
    return [modules[x] for x in module_list]


def sortModules(module_list):
    graph = generateTopology(module_list)
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

        modules[name] = AnalyzerModule(name, func, **kwargs)
        return func

    return decorator
