from attrs import define
import abc
import hist
from cattrs.strategies import include_subclasses


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

    def toHist(self):
        a = hist.axis.Integer(self.start, self.stop, name=self.name)
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
class VariableAxis(Axis):
    edges: list[float | tuple[float, float, int]]
    name: str
    unit: str | None = None

    def toHist(self):
        import numpy as np

        def toList(x):
            if isinstance(x,tuple):
                return list(float(y) for y in np.arange(*x))
            else:
                return [x]

        print([y for x in self.edges for y in toList(x)])
        a = hist.axis.Variable(
            [y for x in self.edges for y in toList(x)], name=self.name
        )
        if self.unit:
            a.unit = self.unit
        return a

def configureConverter(conv):
    # union_strategy = ft.partial(configure_tagged_union, tag_name="module_name")
    include_subclasses(Axis, conv)  # , union_strategy=union_strategy)
