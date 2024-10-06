from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from ..utils import doFormatting
from analyzer.postprocessing.style import Styler
from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .utils import saveFig

def plot2D(
    histogram_name,
    sector,
    output_name,
    style_set,
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    #fig, ax = plt.subplots()
    p = sector.sector_params
    style = styler.getStyle(p)
    h = sector.histograms[histogram_name].get()
    art,_,_ = h.plot2d_full()
    ax = art.pcolormesh.axes
    fig = ax.get_figure()
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    addCMSBits(ax, [sector])
    o = doFormatting(output_name, p, histogram_name=histogram_name)
    saveFig(fig, o)
    plt.close(fig)
