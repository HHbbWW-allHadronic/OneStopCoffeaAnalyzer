import numpy as np
from rich import print

import operator as op
from collections import defaultdict
import functools as ft
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
from analyzer.postprocessing.style import Styler

# from ..grouping import doFormatting
from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .mplstyles import loadStyles
from .utils import addAxesToHist, fixBadLabels, saveFig, scaleYAxis
from analyzer.utils.debugging import jumpIn


def getRatioAndUnc(num, den, uncertainty_type="poisson-ratio"):
    import hist.intervals as hinter

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = num / den
        unc = hinter.ratio_uncertainty(
            num=num, denom=den, uncertainty_type=uncertainty_type
        )
    return ratios, unc


def plotOne(
    histograms,
    stacked_hists,
    common_metadata,
    output_path,
    style_set,
    scale="linear",
    normalize=False,
    plot_configuration=None,
):
    stacked_hists = stacked_hists or []
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    fig, ax = plt.subplots()
    h = None
    for item, meta in histograms:
        title = meta["dataset_title"]
        h = item.histogram
        style = styler.getStyle(meta)
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            yerr=style.yerr,
            flow="none",
            **style.get(),
        )
    if h is None:
        h = stacked_hists[0]
    if stacked_hists:
        stacked_hists = sorted(
            stacked_hists, key=lambda x: x.item.histogram.sum().value
        )
        style_kwargs = defaultdict(list)
        hists = []
        titles = []
        for item, meta in stacked_hists:
            hists.append(item.histogram)
            titles.append(meta["dataset_title"])
            style = styler.getStyle(meta)
            for k, v in style.get().items():
                style_kwargs[k].append(v)

        style_kwargs["histtype"] = style_kwargs["histtype"][0]

        mplhep.histplot(
            hists,
            ax=ax,
            stack=True,
            **style_kwargs,
            label=titles,  # sort="yield"
        )

    labelAxis(ax, "y", h.axes, label=pc.y_label)
    labelAxis(ax, "x", h.axes, label=pc.x_label)
    addCMSBits(
        ax,
        [x.metadata for x in histograms] + [x.metadata for x in stacked_hists],
        extra_text=f"{common_metadata['pipeline']}",
        plot_configuration=pc,
    )
    if style.legend:
        legend_kwargs = {}
        if style.legend_font:
            legend_kwargs["fontsize"] = style.legend_font
        ax.legend(loc="upper right", **legend_kwargs)
        mplhep.sort_legend(ax=ax)
    ax.set_yscale(scale)
    mplhep.yscale_legend(ax, soft_fail=True)
    # mplhep.yscale_anchored_text(ax, soft_fail=True)
    if style.y_min:
        ax.set_ylim(bottom=style.y_min)
    else:
        mplhep.ylow(ax)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def makeStrHist(data, ax_name=None):
    import hist

    ax = hist.axis.StrCategory([x[0] for x in data], name=ax_name)
    h = hist.Hist(ax, storage="double")
    h[:] = np.array([x[1] for x in data])
    return h


def __plotStrCatOne(
    getter,
    sectors,
    output_path,
    style_set,
    ax_name=None,
    normalize=False,
    scale="linear",
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    loadStyles()
    mpl.use("Agg")

    fig, ax = plt.subplots(layout="constrained")
    for sector in sectors:
        p = sector.sector_params
        style = styler.getStyle(p)
        h = makeStrHist(getter(sector), ax_name=ax_name)
        h.plot1d(
            ax=ax,
            label=sector.sector_params.dataset.title,
            density=normalize,
            **style.get(),
        )
    ax.legend()
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    ax.tick_params(axis="x", rotation=90)
    addCMSBits(
        ax,
        [x.sector_params for x in sectors],
        plot_configuration=pc,
    )
    ax.set_yscale(scale)
    # mplhep.yscale_legend(ax, soft_fail=True)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    # mplhep.yscale_legend(ax, soft_fail=True)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


# def __plotStrCatAsTable(
#     getter,
#     sectors,
#     group_params,
#     output_name,
#     style_set,
#     ax_name=None,
#     normalize=False,
#     plot_configuration=None,
# ):
#     pc = plot_configuration or PlotConfiguration()
#     styler = Styler(style_set)
#     loadStyles()
#     mpl.use("Agg")
#
#     rep_data = getter(sectors[0])
#     col_labels = [x[0] for x in rep_data]
#     rows = []
#     row_labels = []
#
#     figsize = (len(rep_data) * 0.3, len(sectors) * 0.3)
#     fig, ax = plt.subplots(figsize=figsize, layout="constrained")
#     fig.patch.set_visible(False)
#     ax.axis("off")
#     ax.axis("tight")
#     for sector in sectors:
#         p = sector.sector_params
#         styler.getStyle(p)
#         data = getter(sector)
#         row_labels.append(sector.sector_params.dataset.title)
#         rows.append([x[1] for x in data])
#
#     table = ax.table(
#         cellText=rows,
#         rowLabels=row_labels,
#         colLabels=col_labels,
#     )
#
#     o = doFormatting(output_name, group_params, histogram_name=(ax_name or ""))
#     saveFig(fig, o, extension=pc.image_type)
#     plt.close(fig)


def plotStrCat(plot_type, *args, table_mode=False, weighted=False, **kwargs):
    def makeGetter(n):
        def inner(sec):
            if weighted:
                return getattr(sec.result.selection_flow, n)
            else:
                return getattr(sec.result.raw_selection_flow, n)

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
    no_stack=False,
    ratio_hlines=(1.0,),
    ratio_height=0.3,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    gs_kw = dict(height_ratios=[1, ratio_height])

    fig, (ax, ratio_ax) = plt.subplots(2, 1, sharex=True, gridspec_kw=gs_kw)
    # ratio_ax = addAxeshToHist(ax, size=ratio_height, pad=0.3)

    if no_stack:
        den_to_plot = sorted(denominator, key=lambda x: x.metadata["title"].lower())
    else:
        den_to_plot = sorted(denominator, key=lambda x: x.item.histogram.sum().value)
    style_kwargs = defaultdict(list)
    hists = []
    titles = []

    den_hist = denominator[0].item.histogram
    x_values = den_hist.axes[0].centers
    left_edge = den_hist.axes.edges[0][0]
    right_edge = den_hist.axes.edges[-1][-1]

    if not no_stack:
        for item, meta in den_to_plot:
            hists.append(item.histogram)
            titles.append(meta["dataset_title"])
            style = styler.getStyle(meta)
            for k, v in style.get().items():
                style_kwargs[k].append(v)
        style_kwargs["histtype"] = style_kwargs["histtype"][0]
        mplhep.histplot(
            hists,
            ax=ax,
            stack=True,
            density=normalize,
            **style_kwargs,
            label=titles,  # sort="yield"
        )
        den_total = ft.reduce(op.add, (x.item.histogram for x in denominator))
    else:
        den_styles = []
        for item, meta in den_to_plot:
            title = meta["dataset_title"]
            h = item.histogram
            style = styler.getStyle(meta)
            den_styles.append(style)
            h.plot1d(
                ax=ax,
                label=title,
                density=normalize,
                yerr=style.yerr,
                flow="none",
                **style.get(),
            )

    all_ratios, all_uncertainties = [], []

    if not no_stack:
        for item, meta in numerators:
            title = meta["dataset_title"]
            h = item.histogram
            fixBadLabels(h)
            style = styler.getStyle(meta)
            n, d = h.values(), den_total.values()

            ratio, unc = getRatioAndUnc(n, d, uncertainty_type=ratio_type)

            if normalize:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = (n / np.sum(n)) / (d / np.sum(d))
            all_ratios.append(ratio)
            all_uncertainties.append(unc)

            if normalize:
                hplot = h / h.sum(flow=False).value
            else:
                hplot = h

            h.plot1d(
                ax=ax,
                label=title,
                density=normalize,
                yerr=True,
                **style.get(),
            )

            ratio[ratio == 0] = np.nan
            ratio[np.isinf(ratio)] = np.nan
            all_opts = {
                **style.get("errorbar", include_type=False),
                **dict(linestyle="none"),
            }
            ratio_ax.errorbar(
                x_values,
                ratio,
                yerr=unc,
                **all_opts,
            )
            # hist.plot.plot_ratio_array(den, ratio, unc, ax=ratio_ax,
    else:
        num = numerators[0]
        h = num.item.histogram
        meta = num.metadata
        title = meta["dataset_title"]
        style = styler.getStyle(meta)
        fixBadLabels(h)
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            yerr=True,
            **style.get(),
        )
        for den, style in zip(den_to_plot, den_styles):
            n, d = h.values(), den.histogram.values()
            ratio, unc = getRatioAndUnc(n, d, uncertainty_type=ratio_type)

            if normalize:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = (n / np.sum(n)) / (d / np.sum(d))
            all_ratios.append(ratio)
            all_uncertainties.append(unc)

            if normalize:
                hplot = h / h.sum(flow=False).value
            else:
                hplot = h

            ratio[ratio == 0] = np.nan
            ratio[np.isinf(ratio)] = np.nan

            all_opts = {
                **style.get("errorbar", include_type=False),
                **dict(linestyle="none"),
            }
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

    addCMSBits(
        ax,
        [x.metadata for x in numerators] + [x.metadata for x in denominator],
        plot_configuration=pc,
    )

    ratio_ax.set_ylabel("Ratio", loc="center")
    # labelAxis(ax, "x", den_hist.axes)

    ax.tick_params(axis="x", which="both", labelbottom=False)

    mplhep.sort_legend(ax=ax)

    ax.set_yscale(scale)
    labelAxis(ratio_ax, "x", den_hist.axes)
    scaleYAxis(ax)

    # mplhep.yscale_legend(ax, soft_fail=True)
    # mplhep.yscale_anchored_text(ax, soft_fail=True)

    # ratio_ax.set_xlabel("Ratio", loc="center")
    # ratio_ax.set_xlabel("HELLO WORLD")
    # fig.tight_layout()

    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)
