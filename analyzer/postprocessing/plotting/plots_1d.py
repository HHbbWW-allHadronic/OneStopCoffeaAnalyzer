import hist.intervals as hinter
import matplotlib as mpl
import numpy as np

from .plottables import FillType


def requireDim(dim):
    def wrapper(func):
        def inner(ax, hist, *args, **kwargs):
            if len(hist.axes) != dim:
                raise ValueError(
                    f"Histogram must have dimension {dim}, found {len(hist.axes)}"
                )
            return func(ax, hist, *args, **kwargs)
        return inner
    return wrapper


def getCenters(vals):
    return (vals[:, 0] + vals[:, 1]) / 2


@requireDim(1)
def drawAsScatter(ax, hist, title=None, style=None, yerr=True, **kwargs):
    style = style or {}
    axis = hist.axes[0]
    edges = np.array(axis)
    x = getCenters(edges)
    y = hist.values()
    yerr = yerr and (hist.variances() is not None)
    s = style.asMplKwargs() if style else {}
    if yerr:
        variances = hist.variances()
        unc = np.sqrt(variances)
        ax.errorbar(
            x,
            y,
            yerr=unc,
            label=title,
            **kwargs,
            fill=s.get("fill"),
            color=s.get("color"),
            linestyle=s.get("linestyle"),
            marker=s.get("marker"),
        )
    else:
        ax.scatter(x, y, label=title, marker="+", **s, **kwargs)
    return ax


@requireDim(1)
def drawAs1DHist(ax, hist, title=None, style=None, yerr=True, orient="h", **kwargs):
    style = style or {}

    axis = hist.axes[0]
    edges = np.array(axis)
    x = getCenters(edges)
    y = hist.values()
    yerr = yerr and (hist.variances() is not None)
    raw_vals = hist.values()
    vals = raw_vals

    s = style.asMplKwargs() if style else {}
    common = dict(
        label=title,
        **kwargs,
    )

    side_edges = edges[:, 0]
    side_edges = np.append(side_edges, edges[-1, 1])

    if yerr:
        errs = np.sqrt(hist.variances())
        if orient == "h":
            ax.errorbar(
                x,
                raw_vals,
                yerr=errs,
                fmt="none",
                fill=s.get("fill"),
                color=s.get("color"),
                **kwargs,
            )
        else:
            ax.errorbar(
                raw_vals,
                x,
                xerr=errs,
                fmt="none",
                fill=s.get("fill"),
                color=s.get("color"),
                **kwargs,
            )

    ax.stairs(
        raw_vals,
        side_edges,
        orientation="vertical" if orient == "h" else "horizontal",
        label=title,
        fill=s.get("fill"),
        color=s.get("color"),
        linewidth=s.get("linewidth", mpl.rcParams["lines.linewidth"]),
        **kwargs,
    )

    return ax


# @requireDim(1)
# def drawRatio(ax, hist):
#     ax.axhline(y=1, linestyle="--", linewidth="1", color="k")
#     hline_list = hline_list or []
#     nv, dv = numerator.values(), denominator.values()
#     n, d = nv / weights[0], dv / weights[1]
#     with np.errstate(divide="ignore", invalid="ignore"):
#         ratio = np.divide(nv, dv, out=np.ones_like(nv), where=(dv != 0))

#     unc = hinter.ratio_uncertainty(
#         n.astype(int),
#         d.astype(int),
#         uncertainty_type=uncertainty_type,
#     ) * (weights[0] / weights[1])

#     x = numerator.axes[0].centers

#     ax.errorbar(
#         x,
#         ratio,
#         yerr=unc,
#         marker="_",
#         linestyle="none",
#         **kwargs,
#     )
#     return ax


# def drawPull(ax, pred, obs, uncertainty_type="poisson", hline_list=None, **kwargs):
#     hline_list = hline_list or []
#     ov, pv = obs.values(), pred.values()
#     unc = np.sqrt(pred.variances())
#     ounc = np.sqrt(obs.variances())
#     # real_unc = np.sqrt(unc**2 + ounc**2)
#     real_unc = ounc

#     pull = np.divide(
#         ov - pv,
#         real_unc,
#         out=np.zeros_like(real_unc),
#         where=real_unc != 0,
#     )
#     x = obs.axes[0].centers

#     if pred.mask is not None:
#         x = x[pred.mask]
#         pull = pull[pred.mask]

#     ax.plot(
#         x,
#         pull,
#         marker="o",
#         markersize=2.5,
#         linestyle="none",
#         **kwargs,
#     )
#     for y in hline_list:
#         ax.axhline(y, linewidth=1, linestyle="--", color="tab:grey")
#     return ax


def setAxisTitles1D(
    ax, axis, y_label=None, y_label_complete=None, x_label=None, top_pad=0.3
):
    if y_label and y_label_complete:
        raise ValueError(f"Can only specify one of y_label or y_label_complete")
    x_unit = getattr(axis, "unit", None)
    if not x_label:
        x_label = axis.name
        if x_unit:
            x_label += f" [{x_unit}]"
    bottom_axes = getattr(ax, "bottom_axes", None)
    ax.set_xlabel(x_label)

    if y_label_complete:
        y_label = y_label_complete
    else:
        y_label = y_label or "Events"
        if x_unit:
            y_label += f" / {x_unit}"
    ax.set_ylabel(y_label)

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
