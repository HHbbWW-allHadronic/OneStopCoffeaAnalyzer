from attrs import define
import abc
import hist

@define
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
        return hist.axis.Integer(self.start, self.stop, name=self.name)


@define(frozen=True)
class RegularAxis(Axis):
    bins: int
    start: float
    stop: float
    name: str
    unit: str | None = None

    def toHist(self):
        return hist.axis.Regular(self.bins, self.start, self.stop, name=self.name)
