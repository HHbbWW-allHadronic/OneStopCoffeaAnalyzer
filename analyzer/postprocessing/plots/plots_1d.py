from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import mplhep

from .plotting import (
    autoScale,
    getRatioAndUnc,
    labelAxis,
)
from .plotting.utils import addAxesToHist
from analyzer.postprocessing.style import Styler

def _plotOne(
    histogram,
    sectors,
    output_name,
    style_set,
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
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
    autoScale(ax)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


def _plotRatio(
    histogram,
    numerators,
    denominator,
    output_name,
    style_set,
    normalize=False,
    middle=1,
    y_lim=None,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    den = denominator.histograms[histogram].get()
    addAxesToHist(ax, num_bottom=1)
    mplhep.histplot(
        den,
        ax=ax,
        label=denominator.sector_params.dataset.title,
        density=normalize,
        **style.get("step"),
    )
    all_r, all_u = [], []
    for sector in numerators:
        p = sector.sector_params
        style = styler.getStyle(p)
        h = sector.histograms[histogram].get()
        ratio, unc = getRatioAndUnc(h.values(), den.values())
        all_r.append(ratio)
        all_u.append(unc)
        s = styler.getStyle(sector.sector_params)
        mplhep.histplot(
            h,
            ax=ax,
            label=sector.sector_params.dataset.title,
            density=normalize,
            **s.get("step"),
        )

        mpl.histplot(
            ratio,
            bins=h.axes[0],
            yerr=unc,
            ax=ax.bottom_axes[0],
            style=s.get("errorbar"),
        )
    all_r = np.concatenate(all_r).flatten()
    all_u = np.concatenate(all_u, axis=1)
    if y_lim is None:
        valid_ratios_idx = np.where(~np.isnan(all_r))
        valid_ratios = all_r[valid_ratios_idx]
        extrema = np.array(
            [
                valid_ratios - all_u[0][valid_ratios_idx],
                valid_ratios + all_u[1][valid_ratios_idx],
            ]
        )
        max_delta = np.amax(np.abs(extrema - middle))
        ratio_extrema = np.abs(max_delta + middle)
        _alpha = 2.0
        scaled_offset = max_delta + (max_delta / (_alpha * ratio_extrema))
        y_lim = [middle - scaled_offset, middle + scaled_offset]

    labelAxis(ax, "y", den.axes)
    labelAxis(ax.bottom_axes[0], "x", den.axes)
    autoScale(ax)
    addCMSBits(ax, (denominator,))
    ax.bottom_axes[0].set_ylim([0, 2])
    ax.bottom_axes[0].set_ylabel("Ratio")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    fig.tight_layout()
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


def _drawCutflow(
    sectors,
    output_name,
    style_set,
    mode="cutflow",
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
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
    autoScale(ax)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


def _drawCutflow(
    sectors,
    output_name,
    style_set,
    mode="cutflow",
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
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
    autoScale(ax)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)
