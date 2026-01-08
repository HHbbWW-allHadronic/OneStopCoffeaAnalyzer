from __future__ import annotations

import functools as ft
from typing import Literal
import itertools as it
from .plots.common import PlotConfiguration
from .style import StyleSet
from analyzer.utils.querying import BasePattern, Pattern, gatherByCapture, NO_MATCH
from analyzer.utils.structure_tools import (
    deepWalkMeta,
    SimpleCache,
    ItemWithMeta,
    commonDict,
    dictToDot,
    doFormatting,
)
from .processors import BasePostprocessor
from .plots.plots_1d import plotOne, plotRatio
from .plots.plots_2d import plot2D
from .grouping import GroupBuilder
from analyzer.utils.structure_tools import globWithMeta
from attrs import define, field
import abc
import pickle as pkl
from pathlib import Path
import lz4.frame


def exportItem(
    item,
    meta,
    output_path,
    compressed=True,
):
    ret = {"metadata": meta, "item": item}
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if compressed:
        with lz4.frame.open(output_path, "wb") as f:
            pkl.dump(ret, f)
    else:
        with open(output_path, "wb") as f:
            pkl.dump(ret, f)


@define
class Dump(BasePostprocessor):
    output_name: str
    compressed: bool = True

    def getRunFuncs(self, group, prefix=None):
        if len(group) != 1:
            raise RuntimeError()
        item, meta = group[0]
        output_path = doFormatting(self.output_name, **dict(meta), prefix=prefix)
        yield ft.partial(
            exportItem, item.histogram, meta, output_path, compressed=self.compressed
        )


def writeNumpy(path, data):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "wb") as f:
        np.save(f, data)


@define
class DumpNPZ(BasePostprocessor):
    output_name: str

    def getRunFuncs(self, group, prefix=None):
        if len(group) != 1:
            raise RuntimeError()
        item, meta = group[0]

        output_path = doFormatting(self.output_name, **dict(meta), prefix=prefix)
        yield ft.partial(writeNumpy, item.saved_columns, meta, output_path)
