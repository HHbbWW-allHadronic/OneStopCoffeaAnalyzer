import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from analyzer.postprocessing.style import Styler

from ..grouping import doFormatting
from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .utils import saveFig
from .mplstyles import loadStyles


def plot2D(
    histogram_name,
    group_params,
    sector,
    output_name,
    style_set,
    normalize=False,
    plot_configuration=None,
    color_scale="linear",
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    loadStyles()
    # fig, ax = plt.subplots()
    p = sector.sector_params
    style = styler.getStyle(p)
    h = sector.histograms[histogram_name].get()
    if normalize:
        h = h / np.sum(h.values())
    if color_scale == "log":
        art = h.plot2d(norm=matplotlib.colors.LogNorm())
    else:
        art = h.plot2d()
    ax = art.pcolormesh.axes
    fig = ax.get_figure()
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    addCMSBits(
        ax,
        [sector],
        extra_text=f"{p.region_name}\n{p.dataset.title}",
        text_color="white",
        plot_configuration=plot_configuration,
    )
    o = doFormatting(output_name, group_params, histogram_name=histogram_name)
    saveFig(fig, o, extension=plot_configuration.image_type)
    plt.close(fig)
