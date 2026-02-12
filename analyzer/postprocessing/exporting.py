from __future__ import annotations

import functools as ft
from analyzer.utils.structure_tools import dotFormat, dictToDot
from .processors import BasePostprocessor
from attrs import define
import pickle as pkl
from pathlib import Path
import lz4.frame
import numpy as np


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
        output_path = dotFormat(self.output_name, **dict(meta), prefix=prefix)
        yield ft.partial(
            exportItem,
            item.histogram,
            dict(meta),
            output_path,
            compressed=self.compressed,
        )


def writeNumpy(path, data, order, mask_fill_value=0):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    def fill(x):
        if isinstance(x, np.ma.MaskedArray):
            return x.filled(mask_fill_value)
        return x

    data = np.stack([fill(data[key]) for key in order], axis=1)
    np.save(path, data)


@define
class DumpNumpy(BasePostprocessor):
    output_name: str
    order: list[str]
    mask_fill_value: float = 0

    def getRunFuncs(self, group, prefix=None):
        if len(group) != 1:
            raise RuntimeError()
        item, meta = group[0]

        output_path = dotFormat(
            self.output_name, **dict(dictToDot(meta)), prefix=prefix
        )
        yield ft.partial(
            writeNumpy,
            output_path,
            item.data,
            self.order,
            mask_fill_value=self.mask_fill_value,
        )
