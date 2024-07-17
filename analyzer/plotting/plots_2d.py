from dataclasses import dataclass

import matplotlib.pyplot as plt
from .plottables import FillType
import numpy as np
from matplotlib.collections import PatchCollection, RegularPolyCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import addAxesToHist


def drawAs2DHist(ax, plot_object, divider=None, add_color_bar=True, **kwargs):
    a1, a2 = plot_object.axes
    ex, ey = a1.flat_edges, a2.flat_edges
    vals = plot_object.values()
    vx, vy = np.meshgrid(ex, ey)
    im = ax.pcolormesh(vx, vy, vals.T, **kwargs)
    ax.set_xlabel(a1.title)
    ax.set_ylabel(a2.title)
    ax.quadmesh = im
    if divider is None:
        divider = make_axes_locatable(ax)
    if add_color_bar:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax.quadmesh, cax=cax)
        cax.get_yaxis().set_offset_position("left")
        ax.cax = cax
    return ax


def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_margin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta * left
    right = lim[1] + delta * right
    ax.set_xlim(left, right)


def addTitles2D(ax, plot_object):
    values = plot_object.values()
    axes = plot_object.axes
    x_unit = getattr(axes[0], "unit", None)
    y_unit = getattr(axes[1], "unit", None)
    ax.set_xlabel(axes[0].title + (f" [{x_unit}]" if x_unit else ""))
    ax.set_ylabel(axes[1].title + (f" [{y_unit}]" if y_unit else ""))

    zlab = FillType.getAxisTitle(plot_object.fill_type)

    if x_unit and y_unit and x_unit == y_unit:
        zlab += f" / {x_unit or ''}$^2$"
    elif x_unit or y_unit:
        zlab += f" / {x_unit or ''} {y_unit or ''}"

    if hasattr(ax, "cax"):
        cax = ax.cax
        cax.set_ylabel(zlab)

    return ax


def drawAs2DExtended(
    ax,
    plot_object_2d,
    top_stack=None,
    right_stack=None,
    color_bar=False,
    top_pad=0.2,
    right_pad=0.2,
    **plotopts,
):
    divider = make_axes_locatable(ax)
    ax = drawAs2DHist(
        ax, plot_object_2d, divider=divider, add_color_bar=False, **plotopts
    )

    ax.top_axes = []
    ax.right_axes = []
    for plot_object in top_stack or []:
        ax_histx = divider.append_axes("top", 1, pad=top_pad, sharex=ax)
        drawAs1DHist(ax_histx, plot_object)
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax.top_axes.append(ax_histx)

    for plot_object in right_stack or []:
        ax_histy = divider.append_axes("right", 1, pad=right_pad, sharey=ax)
        drawAs1DHist(ax_histy, plot_object, orient="v")
        ax_histy.yaxis.set_tick_params(labelleft=False)
        ax.right_axes.append(ax_histy)

    if color_bar:
        cax = divider.append_axes("left", size="5%", pad=1.5)
        cbar = plt.colorbar(ax.quadmesh, cax=cax, orientation="vertical")
        ax.cax = cax
    return ax


@dataclass
class AnnotRectangle:
    x: float
    y: float
    w: float
    h: float
    text: str
    value: float


def drawRectangles(ax, rects):
    ymax = max(r.y + r.h for r in rects)
    xmax = max(r.x + r.w for r in rects)
    ymin = min(r.y for r in rects)
    xmin = min(r.x for r in rects)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    patches = [
        Rectangle(
            (r.x, r.y),
            r.w,
            r.h,
        )
        for r in rects
    ]
    col = PatchCollection(patches)
    col.set_array([r.value for r in rects])
    ax.add_collection(col)
    ax.autoscale_view()
    for r in rects:
        ax.annotate(
            r.text,
            (r.x + r.w / 2, r.y + r.h / 2),
            color="w",
            weight="bold",
            fontsize=6,
            ha="center",
            va="center",
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(col, cax=cax)
    cax.get_yaxis().set_offset_position("left")
    ax.cax = cax
    return ax
