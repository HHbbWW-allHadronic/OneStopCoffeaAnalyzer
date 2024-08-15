import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import enum

import hist


class PlotAxis:
    edges: np.typing.NDArray[Any]
    title: Optional[str] = None
    unit: Optional[str] = None

    def __init__(self, edges, title=None, unit=None):
        self.edges = PlotAxis.normalizeEdges(edges)
        self.title = title
        self.unit = unit

    @staticmethod
    def normalizeEdges(edges):
        return edges

    @property
    def centers(self):
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def flat_edges(self):
        return self.edges

    @staticmethod
    def fromHist(hist_axis):
        return PlotAxis(
            hist_axis.edges,
            getattr(hist_axis, "label", None),
            getattr(hist_axis, "unit", None),
        )

    def __str__(self):
        return f'Axis([{self.edges[0]:0.3g},{self.edges[-1]:0.3g}], bins={len(self.edges) - 1}, title="{self.title}" )'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if self.edges.shape != other.edges.shape:
            return False
        return (
            np.allclose(self.edges, other.edges)
            and self.title == other.title
            and self.unit == other.unit
        )


class FillType(enum.Enum):
    NormalizedEvents = enum.auto()
    WeightedEvents = enum.auto()
    UnweightedEvents = enum.auto()
    Ratio = enum.auto()
    Other = enum.auto()

    @staticmethod
    def getAxisTitle(v):
        mapping = {
            FillType.NormalizedEvents: "Normalized Events",
            FillType.WeightedEvents: "Weighted Events",
            FillType.UnweightedEvents: "Events",
            FillType.Ratio: "Ratio",
            FillType.Other: None,
        }
        return mapping[v]


class PlotObject:
    def __init__(
        self,
        values: np.typing.NDArray[Any],
        axes: Tuple[PlotAxis],
        variances: Optional[np.typing.NDArray[Any]] = None,
        title: Optional[str] = None,
        style: Optional[Dict[str, Any]] = None,
        mask: Optional[np.typing.NDArray[Any]] = None,
        fill_type: FillType = FillType.WeightedEvents,
    ):
        self.__values = values
        self.__axes = axes
        self.__variances = variances
        self.title = title
        self.style = style or {}
        self.mask = mask
        self.fill_type = fill_type

    @property
    def axes(self):
        return self.__axes

    def hasFlow(self):
        return self.__values.shape != tuple(len(x.centers) for x in self.axes)

    def values(self, flow=False):
        if self.hasFlow() and not flow:
            cuts = tuple(slice(1, -1) for i in range(self.__values.ndim))
            return self.__values[cuts]
        else:
            return self.__values
    
    def update_values(self,new_values):
        self.__values = new_values

    def sum(self, flow=False):
        return np.sum(self.values(flow))

    def variances(self, flow=False):
        if self.hasFlow() and not flow:
            cuts = tuple(slice(1, -1) for i in range(self.__values.ndim))
            return self.__variances[cuts]
        else:
            return self.__variances

    def dropFlow(self):
        new_vals = self.values(flow=False)
        new_vars = self.variances(flow=False)

        return PlotObject(
            new_vals,
            self.__axes,
            variances=new_vars,
            title=self.title,
            style=self.style,
            mask=self.mask,
            fill_type=self.fill_type,
        )

    def normalize(self):
        total = self.sum()
        new_vals = self.values() / total
        new_vars = self.variances() / (total * total)
        return PlotObject(
            new_vals,
            self.__axes,
            variances=new_vars,
            title=self.title,
            style=self.style,
            mask=self.mask,
            fill_type=FillType.NormalizedEvents,
        )

    @staticmethod
    def fromHist(hist, **kwargs):
        return PlotObject(
            hist.values(flow=True),
            tuple(PlotAxis.fromHist(a) for a in hist.axes),
            variances=hist.variances(flow=True),
            **kwargs,
        )

    @staticmethod
    def fromNumpy(hist, variances=None, **kwargs):
        return PlotObject(
            hist[0],
            tuple(
                PlotAxis(a)
                for a in (hist[1] if isinstance(hist[1], tuple) else [hist[1]])
            ),
            variances=variances,
            **kwargs,
        )

    def compat(self, other):
        return other.axes == self.axes


def createPlotObject(name, hist, manager=None):
    if manager:
        return PlotObject.fromHist(
            hist, title=manager[name].title, style=manager[name].style
        )
    else:
        return PlotObject.fromHist(hist, title=name)
