from pathlib import Path

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
from analyzer.postprocessing.style import Styler

from ..utils import doFormatting
from .annotations import addCMSBits, labelAxis
from .utils import addAxesToHist,  saveFig
from .common import PlotConfiguration


def getRatioAndUnc(num, den, uncertainty_type="poisson-ratio"):
    import hist.intervals as hinter

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = num / den
        unc = hinter.ratio_uncertainty(
            num=num, denom=den, uncertainty_type=uncertainty_type
        )
    return ratios, unc




def plotOne(
    histogram,
    sectors,
    output_name,
    style_set,
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    mpl.use("Agg")
    fig, ax = plt.subplots()
    for sector in sectors:
        p = sector.sector_params
        style = styler.getStyle(p)
        h = sector.histograms[histogram].get()
        h.plot1d(
            ax=ax,
            label=sector.sector_params.dataset.title,
            density=normalize,
            **style.get("step"),
        )
        # mplhep.histplot(
        #     h,
        #     ax=ax,
        #     label=sector.sector_params.dataset.title,
        #     density=normalize,
        #     **style.get("step"),
        # )

    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


def plotRatio(
    histogram,
    numerators,
    denominator,
    output_name,
    style_set,
    normalize=False,
    middle=1,
    ratio_ylim=(0, 2),
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    mpl.use("Agg")
    fig, ax = plt.subplots()
    den = denominator.histograms[histogram].get()
    ratio_ax = addAxesToHist(ax, size=1.5, pad=0.3)
    p = denominator.sector_params
    style = styler.getStyle(p)
    mplhep.histplot(
        den,
        ax=ax,
        label=denominator.sector_params.dataset.title,
        density=normalize,
        **style.get("step"),
    )
    x_values = den.axes[0].centers
    left_edge = den.axes.edges[0][0]
    right_edge = den.axes.edges[-1][-1]

    all_ratios, all_uncertainties = [], []
    for sector in numerators:
        p = sector.sector_params
        s = styler.getStyle(sector.sector_params)
        h = sector.histograms[histogram].get()
        n, d = h.values(), den.values()
        ratio, unc = getRatioAndUnc(n, d)

        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (n / np.sum(n)) / (d / np.sum(d))

        all_ratios.append(ratio)
        all_uncertainties.append(unc)

        mplhep.histplot(
            h,
            ax=ax,
            label=sector.sector_params.dataset.title,
            density=normalize,
            **s.get("step"),
        )

        ratio[ratio == 0] = np.nan
        ratio[np.isinf(ratio)] = np.nan


        all_opts = {**s.get("errorbar"), **dict(linestyle="none")}
        ratio_ax.errorbar(
            x_values,
            ratio,
            yerr=unc,
            **all_opts,
        )
        # hist.plot.plot_ratio_array(den, ratio, unc, ax=ratio_ax,

    central_value_artist = ratio_ax.axhline(
        middle, color="black", linestyle="dashed", linewidth=1.0
    )
    ratio_ax.set_xlim(left_edge, right_edge)
    ratio_ax.set_ylim(bottom=ratio_ylim[0], top=ratio_ylim[1])
    if normalize:
        y_label = "Normalized Events"
    else:
        y_label = None

    labelAxis(ax, "y", den.axes, label=y_label)
    ax.legend(loc="upper right")
    ax.set_xlabel(None)
    labelAxis(ratio_ax, "x", den.axes)
    addCMSBits(ax, (denominator,))  #
    ratio_ax.set_ylabel("Ratio", loc="center")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    fig.tight_layout()
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


def drawCutflow(
    sectors,
    output_name,
    style_set,
    mode="cutflow",
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    mpl.use("Agg")
    fig, ax = plt.subplots()
    for sector in sectors:
        p = sector.sector_params
        style = styler.getStyle(p)
        cutflow = sector.cutflow_data.cutflow.cutflow
        data = getattr(sector.cutflow_data.cutflow, mode)
        vals = [x[1] for x in data]
        bins = [x[0] for x in data]
        mplhep.histplot(
            vals,
            bins=bins,
            ax=ax,
            label=sector.sector_params.dataset.title,
            density=normalize,
            **style.get("step"),
        )

    ax.set_xlabel("Cut")
    ax.set_ylabel("Events")
    labelAxis(ax, "x", h.axes)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


def drawCutflow(
    sectors,
    output_name,
    style_set,
    mode="cutflow",
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    mpl.use("Agg")
    fig, ax = plt.subplots()
    for sector in sectors:
        p = sector.sector_params
        style = styler.getStyle(p)
        cutflow = sector.cutflow_data.cutflow.cutflow
        data = getattr(sector.cutflow_data.cutflow, mode)
        vals = [x[1] for x in data]
        bins = [x[0] for x in data]
        mplhep.histplot(
            vals,
            bins=bins,
            ax=ax,
            label=sector.sector_params.dataset.title,
            density=normalize,
            **style.get("step"),
        )

    ax.set_xlabel("Cut")
    ax.set_ylabel("Events")
    labelAxis(ax, "x", h.axes)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)
