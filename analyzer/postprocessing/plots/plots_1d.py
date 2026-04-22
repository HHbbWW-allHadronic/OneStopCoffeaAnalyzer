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
from .utils import saveFig, scaleYAxis, addLegend

def hist_sum(h):
    s = h.sum()
    return s.value if hasattr(s, 'value') else s

def get_yerr(h):
    vals = h.values()
    vars = h.variances()
    if vars is None:
        yerr = np.sqrt(vals)
    else:
        yerr = np.sqrt(vars)
    zero_mask = np.isclose(vals, 0.0, atol=1e-10)
    yerr[zero_mask] = 0
    return yerr

def getRatioAndUnc(num, den, uncertainty_type="poisson-ratio"):
    import hist.intervals as hinter

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = num / den
        unc = hinter.ratio_uncertainty(
            num=num, denom=den, uncertainty_type=uncertainty_type
        )
    return ratios, unc

def getYMin(histograms, stacked_hists):
    all_values = []
    for item, _ in histograms:
        all_values.append(item.histogram.values())
    for item, _ in stacked_hists:
        all_values.append(item.histogram.values())

    all_values = np.concatenate([v.flatten() for v in all_values])
    nonzero = all_values[all_values > 0]
    if len(nonzero) == 0:
        return None
    return nonzero.min()

def plotOne(
    histograms,
    stacked_hists,
    common_metadata,
    output_path,
    style_set,
    scale="linear",
    normalize=False,
    show_info=False,
    plot_configuration=None,
    show_stacked_unc=True,
):
    stacked_hists = stacked_hists or []
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    fig, ax = plt.subplots()
    h = None
    if stacked_hists:
        stacked_hists = sorted(
            stacked_hists, key=lambda x: hist_sum(x.item.histogram)
        )
        style_kwargs = defaultdict(list)
        hists = []
        titles = []
        for item, meta in stacked_hists:
            hists.append(item.histogram)
            title = meta.get("title") or meta["dataset_title"]
            if show_info:
                integral = hist_sum(item.histogram)
                counts = item.histogram.values()
                centers = item.histogram.axes[0].centers
                mean = np.average(centers, weights=counts)
                std = np.sqrt(np.average((centers - mean)**2, weights=counts))
                title = f"{title}, Int.={integral:.1f}\nmean={mean:.3f}, std={std:.3f}"
            titles.append(title)
            style = styler.getStyle(meta)
            for k, v in style.get(plottype="fill").items():
                style_kwargs[k].append(v)

        style_kwargs["histtype"] = style_kwargs["histtype"][0]

        mplhep.histplot(
            hists,
            ax=ax,
            stack=True,
            **style_kwargs,
            label=titles,  # sort="yield"
        )
        if show_stacked_unc:
            stacked_total = ft.reduce(op.add, [x.item.histogram for x in stacked_hists])
            yerr = get_yerr(stacked_total)
            
            mplhep.histplot(
                stacked_total,
                ax=ax,
                label="Stacked Unc.",
                histtype="band",
                yerr=yerr
            )

    for item, meta in histograms:
        title = meta.get("title") or meta["dataset_title"]
        h = item.histogram
        if show_info:
            integral = hist_sum(h)
            counts = h.values()
            centers = h.axes[0].centers
            mean = np.average(centers, weights=counts)
            std = np.sqrt(np.average((centers - mean)**2, weights=counts))
            title = f"{title}, Int.={integral:.1f}\nmean={mean:.3f}, std={std:.3f}"
        style = styler.getStyle(meta)
        yerr = get_yerr(h) if style.yerr else style.yerr
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            yerr=yerr,
            **style.get(),
        )

    if h is None:
        h = stacked_hists[0]

    labelAxis(ax, "y", h.axes, label=pc.y_label)
    labelAxis(ax, "x", h.axes, label=pc.x_label)
    addCMSBits(
        ax,
        [x.metadata for x in histograms] + [x.metadata for x in stacked_hists],
        extra_text=f"{common_metadata.get('pipeline')}",
        plot_configuration=pc,
    )

    ax.set_yscale(scale)
    if scale == "log":
        # Ensure small signals are appropriately shown on plots
        ymin = getYMin(histograms, stacked_hists)
        if ymin is not None:
            ax.set_ylim(bottom=ymin * 0.1)
    addLegend(ax, pc)

    scaleYAxis(ax)
    # mplhep.yscale_anchored_text(ax, soft_fail=True)
    if style.y_min:
        ax.set_ylim(bottom=style.y_min)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def makeStrHist(data, ax_name):
    import hist

    ax = hist.axis.StrCategory([x[0] for x in data], name=ax_name)
    h = hist.Hist(ax, storage="weight")
    d = np.array([x[1] for x in data])
    h[:] = np.stack([d, d], axis=-1)
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
    show_yields=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    mpl.use("Agg")

    fig, ax = plt.subplots(layout="constrained")
    for item, meta in items:
        title = meta.get("title") or meta["dataset_title"]
        flow = getter(item)
        style = styler.getStyle(meta)
        h = makeStrHist([(x, y) for x, y in flow.items()], ax_name=ax_name)
        if normalize: 
            h *= (1 / h[0].value)
        if show_yields and normalize:
            values = h.values()
            errs = np.sqrt(h.variances())
            edges = h.axes[0].edges
            centers = 0.5 * (edges[:-1] + edges[1:])
            centers, values, errs = centers[1:], values[1:], errs[1:]
            for x, y, e in zip(centers, values, errs):
                if y > 0:
                    txt = ax.text(
                        x, y+e,
                        f"{100*y:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=20
                    )
                    txt._is_yield_label = True
        h.plot1d(
            ax=ax,
            label=title,
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


def plotStackedDenominators(ax, denominators, styler, normalize=False, show_den_unc=True):
    den_to_plot = sorted(denominators, key=lambda x: hist_sum(x.item.histogram))
    
    hists = []
    titles = []
    style_kwargs = defaultdict(list)


    for item, meta in den_to_plot:
        hists.append(item.histogram)
        titles.append(meta.get("title") or meta["dataset_title"])
        style = styler.getStyle(meta)
        for key, value in style.get(plottype="fill").items():
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
    if show_den_unc:
        yerr = get_yerr(den_total)
        mplhep.histplot(
            den_total,
            ax=ax,
            label="Den. Stat. Unc.",
            histtype="band",
            yerr=yerr
        )
    return den_total


def plotUnstackedDenominators(ax, denominators, styler, *, normalize):
    den_to_plot = sorted(
        denominators, key=lambda x: x.metadata.get("title", "").lower()
    )

    den_styles = []
    for item, meta in den_to_plot:
        style = styler.getStyle(meta)
        den_styles.append(style)
        hist = item.histogram
        yerr = get_yerr(hist) if style.yerr else style.yerr
        hist.plot1d(
            ax=ax,
            label=meta.get("title") or meta["dataset_title"],
            density=normalize,
            yerr=yerr,
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
    show_den_unc_ratio=True,
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
            label=meta.get("title") or meta["dataset_title"],
            density=normalize,
            yerr=get_yerr(hist),
            **style.get(),
        )

        plotRatioErrorBars(ratio_ax, x_values, ratio, unc, style)

    if show_den_unc_ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            den_scaled_uncertainties = np.where(
                den_total.values() != 0,
                np.sqrt(den_total.variances()) / den_total.values(),
                np.nan,
            )
            ratio_ax.bar(
                x=den_total.axes[0].centers,
                bottom=np.nan_to_num(1 - den_scaled_uncertainties, nan=0),
                height=np.nan_to_num(2 * den_scaled_uncertainties, nan=0),
                width=np.diff(den_total.axes[0].edges),
                edgecolor="dimgrey",
                hatch="////",
                fill=False,
                lw=0,
            )


def plotSingleNumeratorMultiDen(
    ax,
    ratio_ax,
    numerator,
    den_to_plot,
    den_styles,
    styler,
    normalize,
    ratio_type,
    x_values,
    ratio_func=computeRatio,
):
    hist = numerator.item.histogram

    n_style = styler.getStyle(numerator.metadata)
    hist.plot1d(
        ax=ax,
        label=numerator.metadata.get("title") or numerator.metadata["dataset_title"],
        density=normalize,
        yerr=get_yerr(hist),
        **n_style.get(),
    )

    for den, style in zip(den_to_plot, den_styles):
        n_vals = hist.values()
        d_vals = den.item.histogram.values()

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
    show_den_unc=True,
    show_den_unc_ratio=False
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
            styler,
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
            show_den_unc=show_den_unc
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
            show_den_unc_ratio=show_den_unc_ratio
        )

    for y in ratio_hlines:
        ratio_ax.axhline(y, color="black", linestyle="dashed", linewidth=1.0)

    ratio_ax.set_xlim(left_edge, right_edge)
    if ratio_ylim is not None:
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
    ax.set_xlabel("")

    addLegend(ax, pc)
    addCMSBits(
        ax,
        [x.metadata for x in numerators] + [x.metadata for x in denominator],
        plot_configuration=pc,
    )

    ax.set_yscale(scale)
    if scale == "log":
        # Ensure small signals are appropriately shown on plots
        ymin = getYMin(numerators, denominator)
        if ymin is not None:
            ax.set_ylim(bottom=ymin * 0.1)

    scaleYAxis(ax)

    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def plotRatioOfRatios(
    num_numerator,
    num_denominator,
    den_numerator,
    den_denominator,
    output_path,
    style_set,
    r1_label="Ratio 1",
    r2_label="Ratio 2",
    double_ratio_label="Ratio of Ratios",
    normalize=False,
    ratio_ylim=(0, 2),
    ratio_type="poisson",
    scale="linear",
    plot_configuration=None,
    ratio_hlines=(1.0,),
    ratio_height=0.3,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    fig, ax, ratio_ax = makeRatioAxes(ratio_height)

    # Assume all share the same axis
    ref_hist = num_denominator.item.histogram
    x_values = ref_hist.axes[0].centers
    left_edge = ref_hist.axes.edges[0][0]
    right_edge = ref_hist.axes.edges[-1][-1]

    # Calculate 4 components
    h_num_num = num_numerator.item.histogram
    h_num_den = num_denominator.item.histogram
    h_den_num = den_numerator.item.histogram
    h_den_den = den_denominator.item.histogram

    # Calculate Ratios
    r1, r1_unc = computeRatio(
        h_num_num.values(),
        h_num_den.values(),
        normalize=normalize,
        ratio_type=ratio_type,
    )

    r2, r2_unc = computeRatio(
        h_den_num.values(),
        h_den_den.values(),
        normalize=normalize,
        ratio_type=ratio_type,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        double_ratio = r1 / r2
        double_ratio_unc = np.abs(double_ratio) * np.sqrt(
            (r1_unc / r1) ** 2 + (r2_unc / r2) ** 2
        )

    double_ratio[np.isnan(double_ratio)] = 0
    double_ratio[np.isinf(double_ratio)] = 0
    double_ratio_unc[np.isnan(double_ratio_unc)] = 0

    style1 = styler.getStyle(num_numerator.metadata)
    style2 = styler.getStyle(den_numerator.metadata)

    opts1 = {
        **style1.get("errorbar", include_type=False),
        "label": r1_label,
    }
    opts2 = {**style2.get("errorbar", include_type=False), "label": r2_label}

    ax.errorbar(x_values, r1, yerr=r1_unc, **opts1)
    ax.errorbar(x_values, r2, yerr=r2_unc, **opts2)

    ratio_ax.errorbar(
        x_values, double_ratio, yerr=double_ratio_unc, color="black", fmt="o"
    )

    for y in ratio_hlines:
        ratio_ax.axhline(y, color="black", linestyle="dashed", linewidth=1.0)

    ratio_ax.set_xlim(left_edge, right_edge)
    ratio_ax.set_ylim(*ratio_ylim)
    ratio_ax.set_ylabel(double_ratio_label)

    labelAxis(ax, "y", ref_hist.axes, label="Ratio")
    labelAxis(ratio_ax, "x", ref_hist.axes)

    addLegend(ax, pc)
    all_meta = [
        x.metadata
        for x in [num_numerator, num_denominator, den_numerator, den_denominator]
    ]
    addCMSBits(ax, all_meta, plot_configuration=pc)

    ax.set_yscale(scale)

    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)
