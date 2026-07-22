from attrs import define
import abc
import hist
import functools as ft
from cattrs.strategies import include_subclasses
from cattrs.strategies import configure_tagged_union


@define(frozen=True)
class Axis(abc.ABC):
    @abc.abstractmethod
    def toHist(self): ...


@define(frozen=True)
class IntegerAxis(Axis):
    name: str
    start: int
    stop: int
    unit: str | None = None
    growth: bool = False

    def toHist(self):
        a = hist.axis.Integer(self.start, self.stop, name=self.name, growth=self.growth)
        if self.unit:
            a.unit = self.unit
        return a


@define(frozen=True)
class RegularAxis(Axis):
    bins: int
    start: float
    stop: float
    name: str = ""
    unit: str | None = None

    def toHist(self):
        a = hist.axis.Regular(self.bins, self.start, self.stop, name=self.name)
        if self.unit:
            a.unit = self.unit
        return a


@define(frozen=True)
class IntCategoryAxis(Axis):
    name: str
    categories: list[int]
    growth: bool = False
    unit: str | None = None

    def toHist(self):
        a = hist.axis.IntCategory(self.categories, name=self.name, growth=self.growth)
        if self.unit:
            a.unit = self.unit
        return a


@define(frozen=True)
class VariableAxis(Axis):
    edges: list[float | tuple[float, float, int]]
    name: str
    unit: str | None = None

    def toHist(self):
        import numpy as np

        def toList(x):
            if isinstance(x, tuple):
                return list(float(y) for y in np.arange(*x))
            else:
                return [x]

        a = hist.axis.Variable(
            [y for x in self.edges for y in toList(x)], name=self.name
        )
        if self.unit:
            a.unit = self.unit
        return a


def configureConverter(conv):
    # union_strategy = ft.partial(configure_tagged_union, tag_name="module_name")
    include_subclasses(
        Axis,
        conv,
        union_strategy=ft.partial(configure_tagged_union, tag_name="module_name"),
    )
    # include_subclasses(Axis, conv)  # , union_strategy=union_strategy)
