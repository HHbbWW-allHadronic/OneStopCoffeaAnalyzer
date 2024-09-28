import pickle as pkl
from dataclasses import dataclass

import hist
import numpy as np

from analyzer.datasets import Style


@dataclass
class HistTarget:
    histogram: hist.Hist
    item: str
    name: str


@dataclass
class FileTarget:
    fname: str
    hname: str
    item: str
    name: str


def mergeHist(axis_name, *hist_targets):
    h_template = hist_targets[0].histogram
    axes = h_template.axes
    items = [x.item for x in hist_targets]
        
    axes.index()

    ret = hist.Hist(*h_template.axes, storage=h_template._storage_type)
    for x in hist_targets:
        val = x.histogram[{axis_name: x.item}].values(flow=True)
        var = x.histogram[{axis_name: x.item}].variances(flow=True)
        if x.item not in ret.axes[axis_name]:
            ret.fill({axis_name: x.item}] = np.stack([val,var], axis=-1)
            
        ret[{axis_name: x.item}] = np.stack([val,var], axis=-1)
        #ret[{axis_name: x.item}] = x.histogram[{axis_name: x.item}].view()
    return ret


def mergeFromFiles(axis_name, *file_targets):
    ht = [
        HistTarget(
            pkl.load(open(ft.fname, "rb"))["histograms"][ft.hname], ft.item, ft.name
        ) for ft in file_targets
    ]
    return mergeHist(axis_name, *ht)
