import itertools as it

import matplotlib.pyplot as plt
import matplotlib as mpl

from .annotations import addEra, addCmsInfo, addCutTable
from .plots_1d import addTitles1D, drawAs1DHist, drawPull, drawRatio, drawAsScatter
from .plots_2d import addTitles2D, drawAs2DHist
from .utils import addAxesToHist


def plotPulls(plotobj_pred, plotobj_obs, coupling, lumi):
    fig, ax = plt.subplots()

    hopo = plotobj_obs
    hppo = plotobj_pred
    ax = drawAs1DHist(hopo, yerr=True, fill=False)

    drawAs1DHist(ax, hppo, yerr=True, fill=False)
    addAxesToHist(ax, num_bottom=1, bottom_pad=0)

    ab = ax.bottom_axes[0]
    drawPull(ab, hppo, hopo)
    ab.set_ylabel(r"$\frac{pred - obs}{\sigma_{pred}}$")
    addEra(ax, lumi or 59.8)
    addCmsInfo(ax, additional_text=f"\n$\\lambda_{{{coupling}}}''$ ")
    addTitles1D(ax, hopo, top_pad=0.2)
    fig.tight_layout()
    return fig


def plotRatio(
    plotobj_pred, plotobj_obs, coupling, lumi, weights=None, no_hists=False, ax=None
):

    hppo = plotobj_pred
    hopo = plotobj_obs

    if not no_hists:
        fig, ax = plt.subplots()

        drawAs1DHist(ax, hopo, yerr=True, fill=False)
        drawAs1DHist(ax, hppo, yerr=True, fill=False)

    addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    ab = ax.bottom_axes[0]
    drawRatio(ab, numerator=hppo, denominator=hopo, weights=weights)

    ab.set_ylabel("Ratio")
    ab.set_ylim(0.6, 1.3)
    # addCmsInfo(ax, additional_text=f"\n$\\lambda_{{{coupling}}}''$ ")
    # addTitles1D(ax, hopo, top_pad=0.2)

    if no_hists:
        return ax
    else:
        fig.tight_layout()
        return fig


def plot1D(
    signal_plobjs,
    background_plobjs,
    lumi,
    coupling,
    era,
    sig_style="hist",
    scale="log",
    xlabel_override=None,
    add_label=None,
    top_pad=0.4,
    ratio=False,
    energy="13 TeV",
    control_region=False,
    weights=None,
    cut_table=None,
    cut_list=None,
):
    if cut_table:
        plt.rcParams["figure.figsize"] = (10, 13)
        fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.25]})
        ax = axes[0]
        ax_table = axes[1]
        ax_table.grid(False)
        ax_table.set_axis_off()
    else:
        fig, ax = plt.subplots()
    for o in background_plobjs:
        drawAs1DHist(ax, o, yerr=False)
    for o in signal_plobjs:
        # drawAs1DHist(ax, o, yerr=False)
        if sig_style == "scatter":
            drawAsScatter(ax, o, yerr=True)
        elif sig_style == "hist":
            drawAs1DHist(ax, o, yerr=True, fill=False)
    plobjslist = list(signal_plobjs)
    if ratio:
        if weights is None:
            weights = [1, 1]
        plotRatio(
            plobjslist[0],
            plobjslist[1],
            coupling,
            lumi,
            no_hists=True,
            ax=ax,
            weights=weights,
        )
    ax.set_yscale(scale)
    addEra(ax, lumi, era, energy=energy)

    if control_region:
        add_text = f"\nCR Selection\n"
    else:
        add_text = f"\n$\\lambda_{{{coupling}}}''$ Selection\n"

    addCmsInfo(
        ax,
        additional_text=add_text
        + (add_label or "")
        + ("\n" + cut_list if cut_list else ""),
    )

    hc = next(it.chain(signal_plobjs, background_plobjs))
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*reversed(sorted(zip(labels, handles), key=lambda t: t[0])))
    extra_legend_args = {}
    extra_legend_args["prop"] = {"size": max(14, min(round(50 / len(labels)), 30))}
    l = ax.legend(handles, labels, loc="upper right", **extra_legend_args)
    w = mpl.rcParams["lines.linewidth"]
    for l in ax.get_legend().legend_handles:
        if isinstance(l, mpl.lines.Line2D):
            l.set_linewidth(w)

    if xlabel_override:
        ax.set_xlabel(xlabel_override)

    addTitles1D(ax, hc, top_pad=top_pad)

    if cut_table:
        addCutTable(ax_table, cut_table)

    if "$p_T ( \sum_{n=1}^" in hc.axes[0].title:
        ax.set_xlim(right=600)
    fig.tight_layout()

    return fig


def plot2D(
    plot_obj,
    lumi,
    coupling,
    era,
    sig_style="hist",
    scale="log",
    add_label=None,
    zscore=False,
    energy="13 Tev",
    control_region=False,
    zscorename="",
    cut_table=None,
    cut_list=None,
):
    if cut_table:
        plt.rcParams["figure.figsize"] = (10, 13)
        fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.25]})
        ax = axes[0]
        ax_table = axes[1]
        ax_table.grid(False)
        ax_table.set_axis_off()
    else:
        fig, ax = plt.subplots()

    energy_map = {
        "2016": "13 TeV",
        "2017": "13 TeV",
        "2018": "13 TeV",
        "2022": "13.6 TeV",
        "2023": "13.6 TeV",
        "2024": "13.6 TeV",
    }
    actual_era = era
    actual_energy = energy

    if zscore:
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.colors import LinearSegmentedColormap

        objtitle = "Z-Score"
        color_min = "#ff0342"  # red
        color_center = "#440154"  # purple
        color_max = "#fde725"  # yellow
        cmap = LinearSegmentedColormap.from_list(
            "cmap_name", [color_min, color_center, color_max]
        )
        drawAs2DHist(ax, plot_obj, cmap=cmap, norm=TwoSlopeNorm(0))
    else:
        objtitle = plot_obj.title
        drawAs2DHist(ax, plot_obj)

        if "/" in era:
            split_eras = era.split("/")
        else:
            split_eras = []

        for single_era in split_eras:
            if single_era in plot_obj.title:
                actual_era = single_era
                break
            else:
                actual_era = era

        if "/" in actual_era:
            actual_energy = energy
        else:
            actual_energy = energy_map[actual_era]

    addEra(ax, lumi, actual_era, energy=actual_energy)
    pos = "in"

    if control_region:
        add_text = f"\nCR Selection\n"
    else:
        add_text = f"\n$\\lambda_{{{coupling}}}''$ Selection\n"
    addCmsInfo(
        ax,
        additional_text=add_text
        + (f"{add_label}" if add_label else "")
        + (f", {objtitle}")
        + ("\n" + cut_list if cut_list else ""),
        pos=pos,
        color="white",
    )

    addTitles2D(ax, plot_obj)

    if cut_table:
        addCutTable(ax_table, cut_table)

    if zscorename and hasattr(ax, "cax"):
        cax = ax.cax
        cax.set_ylabel(zscorename)

    return fig
