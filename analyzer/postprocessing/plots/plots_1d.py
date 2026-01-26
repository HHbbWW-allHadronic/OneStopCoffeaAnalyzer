import numpy as np

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
from .utils import saveFig, scaleYAxis, addLegend


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

    ax.set_yscale(scale)
    addLegend(ax, pc)

    scaleYAxis(ax)
    # mplhep.yscale_anchored_text(ax, soft_fail=True)
    if style.y_min:
        ax.set_ylim(bottom=style.y_min)
    else:
        mplhep.ylow(ax)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def makeStrHist(data, ax_name):
    import hist

    ax = hist.axis.StrCategory([x[0] for x in data], name=ax_name)
    h = hist.Hist(ax, storage="double")
    h[:] = np.array([x[1] for x in data])
    return h


def plotDictAsBars(
    items,
    common_meta,
    output_path,
    getter,
    style_set,
    ax_name=None,
    normalize=False,
    scale="linear",
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    mpl.use("Agg")

    fig, ax = plt.subplots(layout="constrained")
    for item, meta in items:
        title = meta["dataset_title"]
        flow = getter(item)
        style = styler.getStyle(meta)
        h = makeStrHist([(x, y) for x, y in flow.items()], ax_name=ax_name)
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            **style.get(),
        )
    ax.legend()
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    ax.tick_params(axis="x", rotation=90)
    addCMSBits(
        ax,
        [x.metadata for x in items],
        plot_configuration=pc,
    )
    ax.set_yscale(scale)
    addLegend(ax, pc)
    mplhep.sort_legend(ax=ax)
    scaleYAxis(ax)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def makeRatioAxes(ratio_height):
    gs_kw = dict(height_ratios=[1, ratio_height])
    fig, (ax, ratio_ax) = plt.subplots(2, 1, sharex=True, gridspec_kw=gs_kw)
    return fig, ax, ratio_ax


def computeRatio(n, d, normalize=False, ratio_type="poisson"):
    ratio, unc = getRatioAndUnc(n, d, uncertainty_type=ratio_type)

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (n / np.sum(n)) / (d / np.sum(d))

    ratio[ratio == 0] = np.nan
    ratio[np.isinf(ratio)] = np.nan
    return ratio, unc


def computeSignificance(n, d, normalize=False, ratio_type="poisson"):
    with np.errstate(divide="ignore", invalid="ignore"):
        significance = n / np.sqrt(d)

    return significance, None


def plotRatioErrorBars(ratio_ax, x_values, ratio, unc, style):
    opts = {
        **style.get("errorbar", include_type=False),
        "linestyle": "none",
    }
    ratio_ax.errorbar(x_values, ratio, yerr=unc, **opts)


def plotStackedDenominators(ax, denominators, styler, normalize=False):
    den_to_plot = sorted(denominators, key=lambda x: x.item.histogram.sum().value)

    hists = []
    titles = []
    style_kwargs = defaultdict(list)

    for item, meta in den_to_plot:
        hists.append(item.histogram)
        titles.append(meta["dataset_title"])
        style = styler.getStyle(meta)
        for key, value in style.get().items():
            style_kwargs[key].append(value)

    style_kwargs["histtype"] = style_kwargs["histtype"][0]

    mplhep.histplot(
        hists,
        ax=ax,
        stack=True,
        density=normalize,
        label=titles,
        **style_kwargs,
    )

    den_total = ft.reduce(op.add, (x.item.histogram for x in denominators))
    return den_total


def plotUnstackedDenominators(ax, denominators, styler, *, normalize):
    den_to_plot = sorted(denominators, key=lambda x: x.metadata["title"].lower())

    den_styles = []
    for item, meta in den_to_plot:
        style = styler.getStyle(meta)
        den_styles.append(style)
        item.histogram.plot1d(
            ax=ax,
            label=meta["dataset_title"],
            density=normalize,
            yerr=style.yerr,
            flow="none",
            **style.get(),
        )

    return den_to_plot, den_styles


def plotMultiNumerators(
    ax,
    ratio_ax,
    numerators,
    den_total,
    styler,
    normalize,
    ratio_type,
    x_values,
    ratio_func=computeRatio,
):
    for item, meta in numerators:
        hist = item.histogram
        style = styler.getStyle(meta)

        n_vals = hist.values()
        d_vals = den_total.values()

        ratio, unc = ratio_func(
            n_vals,
            d_vals,
            normalize=normalize,
            ratio_type=ratio_type,
        )

        hist.plot1d(
            ax=ax,
            label=meta["dataset_title"],
            density=normalize,
            yerr=True,
            **style.get(),
        )

        plotRatioErrorBars(ratio_ax, x_values, ratio, unc, style)


def plotSingleNumeratorMultiDen(
    ax,
    ratio_ax,
    numerator,
    den_to_plot,
    den_styles,
    normalize,
    ratio_type,
    x_values,
    ratio_func=computeRatio,
):
    hist = numerator.item.histogram

    hist.plot1d(
        ax=ax,
        label=numerator.metadata["dataset_title"],
        density=normalize,
        yerr=True,
        **den_styles[0].get(),
    )

    for den, style in zip(den_to_plot, den_styles):
        n_vals = hist.values()
        d_vals = den.histogram.values()

        ratio, unc = ratio_func(
            n_vals,
            d_vals,
            normalize=normalize,
            ratio_type=ratio_type,
        )

        plotRatioErrorBars(ratio_ax, x_values, ratio, unc, style)


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

    fig, ax, ratio_ax = makeRatioAxes(ratio_height)

    den_hist = denominator[0].item.histogram
    x_values = den_hist.axes[0].centers
    left_edge = den_hist.axes.edges[0][0]
    right_edge = den_hist.axes.edges[-1][-1]

    ratio_func = computeSignificance if ratio_type == "significance" else computeRatio
    if no_stack:
        den_to_plot, den_styles = plotUnstackedDenominators(
            ax,
            denominator,
            styler,
            normalize=normalize,
        )
        plotSingleNumeratorMultiDen(
            ax,
            ratio_ax,
            numerators[0],
            den_to_plot,
            den_styles,
            normalize=normalize,
            ratio_type=ratio_type,
            x_values=x_values,
            ratio_func=ratio_func,
        )
    else:
        den_total = plotStackedDenominators(
            ax,
            denominator,
            styler,
            normalize=normalize,
        )
        plotMultiNumerators(
            ax,
            ratio_ax,
            numerators,
            den_total,
            styler,
            normalize=normalize,
            ratio_type=ratio_type,
            x_values=x_values,
            ratio_func=ratio_func,
        )

    for y in ratio_hlines:
        ratio_ax.axhline(y, color="black", linestyle="dashed", linewidth=1.0)

    ratio_ax.set_xlim(left_edge, right_edge)
    ratio_ax.set_ylim(*ratio_ylim)
    if ratio_type == "significance":
        rylabel = "Significance"
    else:
        rylabel = "Ratio"
    ratio_ax.set_ylabel(rylabel)

    labelAxis(
        ax,
        "y",
        den_hist.axes,
        label="Normalized Events" if normalize else None,
    )
    labelAxis(ratio_ax, "x", den_hist.axes)

    addLegend(ax, pc)
    addCMSBits(
        ax,
        [x.metadata for x in numerators] + [x.metadata for x in denominator],
        plot_configuration=pc,
    )

    ax.set_yscale(scale)
    scaleYAxis(ax)

    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)
