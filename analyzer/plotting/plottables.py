import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

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
        if edges.ndim == 1:
            return np.stack((edges[:-1, np.newaxis], edges[1:, np.newaxis]), axis=1)
        return edges

    @property
    def centers(self):
        return np.squeeze((self.edges[:, 0] + self.edges[:, 1]) / 2)

    @property
    def flat_edges(self):
        return np.append(self.edges[:, 0], self.edges[-1, 1])

    @staticmethod
    def fromHist(hist_axis):
        return PlotAxis(
            hist_axis.edges,
            getattr(hist_axis, "label", None),
            getattr(hist_axis, "unit", None),
        )

    def __str__(self):
        return f'Axis([{self.edges[0][0]},{self.edges[-1][1]}], bins={len(self.edges)}, title="{self.title}" )'

    def __repr__(self):
        return str(self)


@dataclass
class PlotObject:
    values: np.typing.NDArray[Any]
    axes: Tuple[PlotAxis]
    variances: Optional[np.typing.NDArray[Any]] = None

    title: Optional[str] = None
    style: Optional[Dict[str, Any]] = None

    mask: Optional[np.typing.NDArray[Any]] = None

    @staticmethod
    def fromHist(hist, title=None, style=None, mask=None):
        return PlotObject(
            values=hist.values(),
            axes=tuple(PlotAxis.fromHist(a) for a in hist.axes),
            variances=hist.variances(),
            title=title,
            style=style or {},
            mask=mask,
        )

    @staticmethod
    def fromNumpy(hist, variances=None, title=None, style=None, mask=None):
        return PlotObject(
            values=hist[0],
            axes=tuple(
                PlotAxis(a)
                for a in (hist[1] if isinstance(hist[1], tuple) else [hist[1]])
            ),
            variances=variances,
            title=title,
            style=style or {},
            mask=mask,
        )


def createPlotObjects(hist, cat_axis, manager, cat_filter=None):
    other_axes = [a for a in hist.axes if a.name != cat_axis]
    titles = [a.label for a in other_axes]
    ret = [
        PlotObject.fromHist(hist[{cat_axis: n}], manager[n].title, manager[n].style)
        for n in hist.axes[cat_axis]
        # if cat_filter is None or (re.search(cat_filter, n))
        if cat_filter is None or cat_filter(n)
    ]
    return ret
