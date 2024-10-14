import pickle as pkl
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from analyzer.postprocessing.style import Styler

from ..utils import doFormatting
from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .utils import addAxesToHist, saveFig


def exportHist(
    histogram_name,
    sectors,
    output_name,
):
    ret = {}
    for sector in sectors:
        p = sector.sector_params
        k = (
            p.dataset.name,
            p.region["region_name"],
        )
        ret[k] = {
            "params": sector.sector_params.model_dump(),
            "hist_name": histogram_name,
            "hist_collection": sector.histograms[histogram_name].model_dump(),
        }
    o = doFormatting(output_name, p, histogram_name=histogram_name)
    Path(o).parent.mkdir(exist_ok=True, parents=True)
    with open(o, "wb") as f:
        pkl.dump(ret, f)
