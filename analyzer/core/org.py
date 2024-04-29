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


class FakeEvents:
    def __init__(self, parent=None):
        self.used_fields = set()
        self.created_fields = set()
        self.parent = parent

    def __setitem__(self, key, val):
        self.created_fields.add(key)

    def __getitem__(self, key):
        if isinstance(key, str):
            self.used_fields.add(key)
        return FakeEvents(self)

    def __getattr__(self, key):
        return self[key]

    def __call__(self, *args, **kwargs):
        return FakeEvents(self)

    def __iter__(self):
        return iter([])

    def __lt__(self, other):
        return FakeEvents(self.parent)

    def __gt__(self, other):
        return FakeEvents(self.parent)

    def __ge__(self, other):
        return FakeEvents(self.parent)

    def __le__(self, other):
        return FakeEvents(self.parent)

    def __and__(self, other):
        return FakeEvents(self.parent)

    def __rand__(self, other):
        return FakeEvents(self.parent)

    def __or__(self, other):
        return FakeEvents(self.parent)

    def __ror__(self, other):
        return FakeEvents(self.parent)

    def __not__(self, other):
        return FakeEvents(self.parent)

    def __invert__(self):
        return FakeEvents(self.parent)

    def __abs__(self):
        return FakeEvents(self.parent)

    def __mul__(self, other):
        return FakeEvents(self.parent)

    def __rmul__(self, other):
        return FakeEvents(self.parent)

    def __div__(self, other):
        return FakeEvents(self.parent)

    def __rdiv__(self, other):
        return FakeEvents(self.parent)

    def __truediv__(self, other):
        return FakeEvents(self.parent)

    def __floordiv__(self, other):
        return FakeEvents(self.parent)

    def __sub__(self, other):
        return FakeEvents(self.parent)

    def __rsub__(self, other):
        return FakeEvents(self.parent)

    def __add__(self, other):
        return FakeEvents(self.parent)

    def __radd__(self, other):
        return FakeEvents(self.parent)


class FakeSelector:
    def __init__(self, parent):
        self.parent = parent

    def add(self, name, *args, **kwargs):
        self.parent.selections.add(name)


class FakeAnalyzer:
    def __init__(self):
        self.created_histograms = set()
        self.selections = set()

        self.selection = FakeSelector(self)

    def H(self, x, *args, **kwargs):
        self.created_histograms.add(x)


class FakeAk:
    def __init__(self, events):
        self.events = events

    def __getattr__(self, key):
        def nothing(*args, **kwargs):
            return self.events

        return nothing


class FakeDecorator:
    def __call__(self, *args, **kwargs):
        def inner(func):
            return func

        return inner


def inspectModule(module):
    module_code = inspect.getsource(module.function)
    fev = FakeEvents()
    fa = FakeAnalyzer()
    code = module_code + f"\n{module.function.__name__}(fev, fa)"
    env = {
        "fev": fev,
        "fa": fa,
        "analyzerModule": FakeDecorator(),
        "ak": FakeAk(fev),
        "np": FakeAk(fev),
    }
    for k, v in module.function.__globals__.items():
        if k in ["analyzerModule", "ak", "np"]:
            continue
        env[k] = v

    exec(code, env)
    return fev.used_fields, fev.created_fields, fa.created_histograms


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
        documentation=None,
    ):
        self.name = name
        self.function = function
        self.depends_on = toSet(depends_on) if depends_on else set()
        self.categories = toSet(categories) if categories else set()
        self.always = always
        self.documenation = documentation

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

        modules[name] = AnalyzerModule(name, func, documentation=func.__doc__, **kwargs)
        return func

    return decorator
