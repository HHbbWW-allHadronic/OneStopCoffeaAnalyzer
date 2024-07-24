import numpy as np

import hist
import hist.intervals as hinter
import matplotlib as mpl
import matplotlib.pyplot as plt
from .plottables import FillType


def drawAsScatter(ax, p, yerr=True, **kwargs):
    style = p.style

    x = p.axes[0].centers
    ed = p.axes[0].flat_edges
    y = p.values()
    if p.mask is not None:
        x = x[p.mask]
        y = y[p.mask]

    if yerr:
        e_start = ed[1:]
        e_end = ed[:-1]

        if p.variances() is None:
            raise ValueError(f"Plot object does not have variance")
        var = np.sqrt(p.variances())
        if p.mask is not None:
            var = var[p.mask]
            e_start = e_start[p.mask]
            e_end = e_end[p.mask]

        ax.errorbar(x, y, yerr=var, fmt="none", **style, **kwargs)
        ax.hlines(y, e_start, e_end, label=p.title, **style)
    else:
        ax.scatter(x, y, label=p.title, marker="+", **style, **kwargs)
    return ax


def drawAs1DHist(ax, plot_object, yerr=True, fill=True, orient="h", **kwargs):
    style = plot_object.style
    x = plot_object.axes[0].centers
    edges = plot_object.axes[0].flat_edges
    raw_vals = plot_object.values()
    vals = np.append(raw_vals, raw_vals[-1])
    w = mpl.rcParams["lines.linewidth"]

    if yerr:
        errs = np.sqrt(plot_object.variances())
        if orient == "h":
            ax.errorbar(
                x, raw_vals, yerr=errs, fmt="none", label=None, **kwargs, **style
            )
        else:
            ax.errorbar(
                raw_vals, x, xerr=errs, fmt="none", label=None, **style, **kwargs
            )
    if fill:
        if orient == "h":
            ax.fill_between(
                edges, vals, step="post", label=plot_object.title, **style, **kwargs
            )
        else:
            ax.fill_betweenx(
                edges, vals, step="post", label=plot_object.title, **style, **kwargs
            )
    else:
        ax.stairs(
            raw_vals,
            edges,
            label=plot_object.title,
            orientation="vertical" if orient == "h" else "horizontal",
            linewidth=w,
            **style,
            **kwargs,
        )

    return ax


def drawRatio(
    ax, numerator, denominator, uncertainty_type="poisson", hline_list=None, **kwargs
):
    hline_list = hline_list or []
    nv, dv = numerator.values(), denominator.values()
    ratio = np.divide(nv, dv, out=np.ones_like(nv), where=dv != 0)

    unc = hinter.ratio_uncertainty(
        numerator.variances(),
        denominator.variances(),
        uncertainty_type=uncertainty_type,
    )
    x = numerator.axes[0].centers
    ax.errorbar(x, ratio, yerr=unc, marker="+", linestyle="none", **kwargs)
    return ax


def drawPull(ax, pred, obs, uncertainty_type="poisson", hline_list=None, **kwargs):
    hline_list = hline_list or []
    ov, pv = obs.values(), pred.values()
    unc = np.sqrt(pred.variances())
    ounc = np.sqrt(obs.variances())
    # real_unc = np.sqrt(unc**2 + ounc**2)
    real_unc = ounc

    pull = np.divide(
        ov - pv,
        real_unc,
        out=np.zeros_like(real_unc),
        where=real_unc != 0,
    )
    x = obs.axes[0].centers

    if pred.mask is not None:
        x = x[pred.mask]
        pull = pull[pred.mask]

    ax.plot(
        x,
        pull,
        marker="o",
        markersize=2.5,
        linestyle="none",
        **kwargs,
    )
    for y in hline_list:
        ax.axhline(y, linewidth=1, linestyle="--", color="tab:grey")
    return ax


def addTitles1D(ax, plot_object, exclude=None, top_pad=0.3):
    axes = plot_object.axes
    unit = getattr(axes[0], "unit", None)
    lab = axes[0].title
    if unit:
        lab += f" [{unit}]"
    bottom_axes = getattr(ax, "bottom_axes", None)
    if bottom_axes:
        bottom_axes[-1].set_xlabel(lab)
    else:
        ax.set_xlabel(lab)
    has_var = plot_object.variances()
    ylab = FillType.getAxisTitle(plot_object.fill_type)
    if unit:
        ylab += f" / {unit}"
    ax.set_ylabel(ylab)

    sc = ax.get_yscale()
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = np.diff(lim)
    if sc == "log":
        top_pad = 10 ** (1 + top_pad)
        bottom = max(1, lim[0] - delta * 0.05)
        bottom = lim[0]
    else:
        bottom = lim[0] - delta * 0.05
    top = lim[1] + delta * top_pad
    ax.set_ylim(bottom, top)
    return ax
