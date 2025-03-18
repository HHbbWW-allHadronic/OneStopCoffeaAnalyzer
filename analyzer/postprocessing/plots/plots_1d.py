import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
from analyzer.postprocessing.style import Styler
from rich import print

from ..grouping import doFormatting
from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .mplstyles import loadStyles
from .utils import addAxesToHist, fixBadLabels, saveFig


def getRatioAndUnc(num, den, uncertainty_type="poisson-ratio"):
    import hist.intervals as hinter

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = num / den
        unc = hinter.ratio_uncertainty(
            num=num, denom=den, uncertainty_type=uncertainty_type
        )
    return ratios, unc


def plotOne(
    packaged_hists,
    group_params,
    output_path,
    style_set,
    scale="linear",
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    fig, ax = plt.subplots()
    for packaged_hist in packaged_hists:
        title = packaged_hist.title
        h = packaged_hist.histogram
        fixBadLabels(h)
        style = styler.getStyle(packaged_hist.sector_parameters)
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            yerr=style.yerr,
            flow="none",
            histtype=style.plottype,
            **style.get(),
        )

    labelAxis(ax, "y", h.axes, label=plot_configuration.y_label)
    labelAxis(ax, "x", h.axes, label=plot_configuration.x_label)
    addCMSBits(
        ax,
        [x.sector_parameters for x in packaged_hists],
        plot_configuration=plot_configuration,
    )
    if style.legend:
        legend_kwargs = {}
        if style.legend_font:
            legend_kwargs["fontsize"] = style.legend_font
        ax.legend(loc="upper right", **legend_kwargs)
        mplhep.sort_legend(ax=ax)
    ax.set_yscale(scale)
    if style.y_min:
        ax.set_ylim(bottom=style.y_min)
    else:
        mplhep.ylow(ax)
    saveFig(fig, output_path, extension=plot_configuration.image_type)
    plt.close(fig)


def makeStrHist(data, ax_name=None):
    import hist

    ax = hist.axis.StrCategory([x[0] for x in data], name=ax_name)
    h = hist.Hist(ax, storage="double")
    h[:] = np.array([x[1] for x in data])
    return h


def __plotStrCatOne(
    getter,
    group_params,
    sectors,
    output_path,
    style_set,
    ax_name=None,
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    print(style_set)
    loadStyles()
    mpl.use("Agg")

    fig, ax = plt.subplots()
    for sector in sectors:
        p = sector.sector_params
        style = styler.getStyle(p)
        print(style)
        h = makeStrHist(getter(sector), ax_name=ax_name)
        h.plot1d(
            ax=ax,
            label=sector.sector_params.dataset.title,
            density=normalize,
            histtype=style.plottype,
            **style.get(),
        )
    ax.legend()
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    ax.tick_params(axis="x", rotation=90)
    addCMSBits(
        ax,
        [x.sector_params for x in sectors],
        plot_configuration=plot_configuration,
    )
    # mplhep.yscale_legend(ax, soft_fail=True)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    # mplhep.yscale_legend(ax, soft_fail=True)
    fig.tight_layout()
    saveFig(fig, output_path, extension=plot_configuration.image_type)
    plt.close(fig)


def __plotStrCatAsTable(
    getter,
    sectors,
    group_params,
    output_name,
    style_set,
    ax_name=None,
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    loadStyles()
    mpl.use("Agg")

    rep_data = getter(sectors[0])
    col_labels = [x[0] for x in rep_data]
    rows = []
    row_labels = []

    figsize = (len(rep_data) * 0.3, len(sectors) * 0.3)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")
    for sector in sectors:
        p = sector.sector_params
        style = styler.getStyle(p)
        data = getter(sector)
        row_labels.append(sector.sector_params.dataset.title)
        rows.append([x[1] for x in data])

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
    )

    o = doFormatting(output_name, group_params, histogram_name=(ax_name or ""))
    fig.tight_layout()
    saveFig(fig, o, extension=plot_configuration.image_type)
    plt.close(fig)


def plotStrCat(plot_type, *args, table_mode=False, **kwargs):
    def makeGetter(n):
        def inner(sec):
            return getattr(sec.result.selection_flow, n)

        return inner

    if table_mode:
        f = __plotStrCatAsTable
    else:
        f = __plotStrCatOne

    f(makeGetter(plot_type), *args, ax_name=plot_type, **kwargs)


def plotRatio(
    denominator,
    numerators,
    output_path,
    style_set,
    normalize=False,
    ratio_ylim=(0, 2),
    ratio_type="poisson",
    scale="linear",
    plot_configuration=None,
    ratio_hlines=(1.0,),
    ratio_height=1.5,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    fig, ax = plt.subplots()
    ratio_ax = addAxesToHist(ax, size=ratio_height, pad=0.3)

    den_hist = denominator.histogram

    fixBadLabels(den_hist)

    style = denominator.style or styler.getStyle(denominator.sector_parameters)
    den_hist.plot1d(
        ax=ax,
        label=denominator.title,
        density=normalize,
        histtype=style.plottype,
        yerr=True,
        **style.get(),
    )

    x_values = den_hist.axes[0].centers
    left_edge = den_hist.axes.edges[0][0]
    right_edge = den_hist.axes.edges[-1][-1]

    all_ratios, all_uncertainties = [], []

    for num in numerators:
        title = num.title
        h = num.histogram
        fixBadLabels(h)
        sp = num.sector_parameters
        s = num.style or styler.getStyle(num.sector_parameters)

        n, d = h.values(), den_hist.values()
        ratio, unc = getRatioAndUnc(n, d, uncertainty_type=ratio_type)
        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (n / np.sum(n)) / (d / np.sum(d))
        all_ratios.append(ratio)
        all_uncertainties.append(unc)
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            yerr=True,
            histtype=s.plottype,
            **s.get(),
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

    for l in ratio_hlines:
        ratio_ax.axhline(l, color="black", linestyle="dashed", linewidth=1.0)

    ratio_ax.set_xlim(left_edge, right_edge)
    ratio_ax.set_ylim(bottom=ratio_ylim[0], top=ratio_ylim[1])
    if normalize:
        y_label = "Normalized Events"
    else:
        y_label = None

    labelAxis(ax, "y", den_hist.axes, label=y_label)
    ax.legend(loc="upper right")
    ax.set_xlabel(None)

    labelAxis(ratio_ax, "x", den_hist.axes)
    addCMSBits(
        ax,
        [denominator.sector_parameters, *(x.sector_parameters for x in numerators)],
        plot_configuration=plot_configuration,
    )

    ratio_ax.set_ylabel("Ratio", loc="center")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    mplhep.sort_legend(ax=ax)
    ax.set_yscale(scale)
    ax.set_yscale(scale)
    fig.tight_layout()
    saveFig(fig, output_path, extension=plot_configuration.image_type)
