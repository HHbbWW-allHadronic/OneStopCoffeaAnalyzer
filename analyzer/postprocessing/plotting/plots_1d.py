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


def drawAsScatter(ax, edges, vals, y_unc=None, title=None, style=None,  **kwargs):
    edges = np.array(edges)
    x = getCenters(edges)
    y = vals
    s = style.asMplKwargs() if style else {}
    if y_unc is not None:
        return ax.errorbar(
            x,
            y,
            yerr=y_unc,
            label=title,
            fill=s.get("fill"),
            color=s.get("color"),
            linestyle="None",
            marker=s.get("marker",'o'),
            markersize=1,
            **kwargs,
        )
    else:
        return ax.scatter(x, y, label=title, marker="+", **s, **kwargs)


def drawAs1DHist(ax, edges, vals, y_unc=None, title=None, style=None,   **kwargs):
    style = style or {}
    edges = np.array(edges)
    x = getCenters(edges)

    s = style.asMplKwargs() if style else {}
    common = dict(
        label=title,
        **kwargs,
    )

    side_edges = edges[:, 0]
    side_edges = np.append(side_edges, edges[-1, 1])
    artists = []

    if y_unc is not None:
        art = ax.errorbar(
            x,
            vals,
            yerr=y_unc,
            fmt="none",
            fill=s.get("fill"),
            color=s.get("color"),
            **kwargs,
        )
        artists.append(art)

    stair = ax.stairs(
        vals,
        side_edges,
        label=title,
        fill=s.get("fill"),
        color=s.get("color"),
        linewidth=s.get("linewidth", mpl.rcParams["lines.linewidth"]),
        **kwargs,
    )
    artists.append(stair)

    return artists


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


def getRatioAndUnc(num,den,uncertainty_type="poisson-ratio"):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(den > 0.0, num / den, 1.0)
    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan

    unc = hinter.ratio_uncertainty(num, den, uncertainty_type=uncertainty_type)
    return ratios, unc

@requireDim(1)
def drawRatio(
    ax,
    numerator,
    denominator,
    title=None,
    style=None,
    uncertainty_type="poisson-ratio",
    uncertainty=True,
        y_lim=None,
    **kwargs,
):
    style = style or {}
    axis = numerator.axes[0]
    edges = np.array(axis)
    x = getCenters(edges)

    num = numerator.values()
    den = denominator.values()
    s = style.asMplKwargs() if style else {}
    ratios,unc=getRatioAndUnc(num,den,uncertainty_type)

    bar_low, bar_high = unc[0], unc[1]
    min_r, max_r = np.min(ratios), np.max(ratios)

    if uncertainty:
        ax.errorbar(
            x,
            ratios,
            yerr=[bar_low, bar_high],
            **kwargs,
            fill=s.get("fill"),
            color=s.get("color"),
            linestyle=s.get("linestyle", None),
            markersize=1,
            marker=s.get("marker", "o"),
        )
    else:
        ax.scatter(
            x,
            ratios,
            fill=s.get("fill"),
            color=s.get("color"),
            linestyle=s.get("linestyle"),
            marker=s.get("marker", "o"),
            **kwargs,
        )

    return ax


def labelAxis(
        ax, which, axes, label=None, label_complete=None
):
    mapping = dict(x=0,y=1,z=2)
    idx = mapping[which]

    if idx  != len(axes):
        this_unit = getattr(axes[idx], "unit", None)
        if not label:
            label = axes[idx].name
            if this_unit:
                label += f" [{this_unit}]"
        getattr(ax,f"set_{which}label")(label)
    else:
        label = label or "Events"
        units = [getattr(x, "unit", None) for x in axes]
        units = [x for x in units if x]
        unit_format = "*".join(units)
        if unit_format:
            label += f" / {unit_format}"
        getattr(ax,f"set_{which}label")(label)

def autoScale(ax, top_pad=0.3):
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

def autoLim(ax, hist):
    ax.set_xlim(hist.axes[0][0], hist.axes[0][-1])

# def setAxisTitles1D(
#     ax, axis, y_label=None, y_label_complete=None, x_label=None, top_pad=0.3
# ):
#     if y_label and y_label_complete:
#         raise ValueError(f"Can only specify one of y_label or y_label_complete")
#     x_unit = getattr(axis, "unit", None)
#     if not x_label:
#         x_label = axis.name
#         if x_unit:
#             x_label += f" [{x_unit}]"
#     bottom_axes = getattr(ax, "bottom_axes", None)
#     ax.set_xlabel(x_label)

#     if y_label_complete:
#         y_label = y_label_complete
#     else:
#         y_label = y_label or "Events"
#         if x_unit:
#             y_label += f" / {x_unit}"
#     ax.set_ylabel(y_label)

#     sc = ax.get_yscale()
#     ax.set_ymargin(0)
#     ax.autoscale_view()
#     lim = ax.get_ylim()
#     delta = np.diff(lim)
#     if sc == "log":
#         top_pad = 10 ** (1 + top_pad)
#         bottom = max(1, lim[0] - delta * 0.05)
#         bottom = lim[0]
#     else:
#         bottom = lim[0] - delta * 0.05
#     top = lim[1] + delta * top_pad
#     ax.set_ylim(bottom, top)
#     return ax
