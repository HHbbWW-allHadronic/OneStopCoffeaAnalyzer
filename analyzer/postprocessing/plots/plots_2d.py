from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import mplhep

from .plotting import (
    autoScale,
    labelAxis,
)
from analyzer.postprocessing.style import Styler
from .annotations import addCMSBits

def _plot2D(
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
    fig, ax = plt.subplots()
    p = sector.sector_params
    style = styler.getStyle(p)
    h = sector.histograms[histogram_name].get()
    mplhep.hist2dplot(
        h,
        ax=ax,
        # label=sector.sector_params.dataset.title,
        density=normalize,
        **style.get("step"),
    )

    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)
