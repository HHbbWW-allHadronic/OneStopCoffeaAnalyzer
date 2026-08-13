from __future__ import annotations
import functools as ft
from typing import Literal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mplhep
from .style import StyleSet
from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    dotFormat,
)
from .processors import BasePostprocessor
from attrs import define, field


def makeDRPlotWithLine(group, common_meta, output_path, style_set, show_info,
                       vline, vline_label, plot_configuration=None):
    from .plots.plots_1d import addCMSBits, addLegend, labelAxis, saveFig, scaleYAxis
    from .plots.common import PlotConfiguration
    from .style import Styler

    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    fig, ax = plt.subplots()

    all_counts = None
    h = None
    for item, meta in group:
        h = item.histogram
        title = meta.get("title") or meta["dataset_title"]
        if show_info:
            integral = h.sum().value
            counts = h.values()
            centers = h.axes[0].centers
            mean = np.average(centers, weights=counts)
            std = np.sqrt(np.average((centers - mean)**2, weights=counts))
            title = f"{title}, Int.={integral:.1f}\nmean={mean:.3f}, std={std:.3f}"
        style = styler.getStyle(meta)
        h.plot1d(ax=ax, label=title, yerr=style.yerr, flow="none", **style.get())
        counts = h.values()
        all_counts = counts if all_counts is None else all_counts + counts

    # Add vertical dashed line
    ax.axvline(x=vline, color="black", linestyle="--", linewidth=1.5,
               label=f"{vline_label} = {vline}")

    # Compute integrals excluding sentinel -1.0 bin
    if h is not None and all_counts is not None:
        centers = h.axes[0].centers
        valid = centers >= 0
        valid_counts = all_counts[valid]
        valid_centers = centers[valid]
        total = np.sum(valid_counts)
        if total > 0:
            left = np.sum(valid_counts[valid_centers < vline])
            right = np.sum(valid_counts[valid_centers >= vline])
            ax.text(0.75, 0.97,
                    f"< {vline}: {100*left/total:.1f}%\n> {vline}: {100*right/total:.1f}%",
                    transform=ax.transAxes, verticalalignment="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    labelAxis(ax, "y", h.axes, label=pc.y_label)
    labelAxis(ax, "x", h.axes, label=pc.x_label)
    addCMSBits(
        ax,
        [x.metadata for x in group],
        extra_text=f"{common_meta.get('pipeline', '')}",
        plot_configuration=pc,
    )
    addLegend(ax, pc)
    scaleYAxis(ax)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


@define
class DRPlotWithLine(BasePostprocessor):
    output_name: str
    style_set: str | StyleSet = field(factory=StyleSet)
    show_info: bool = False
    vline: float = 0.4
    vline_label: str = "threshold"

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeDRPlotWithLine,
            group,
            common_meta,
            output_path,
            style_set=self.style_set,
            show_info=self.show_info,
            vline=self.vline,
            vline_label=self.vline_label,
            plot_configuration=pc,
        )

def make2DDRPlotWithLine(group, common_meta, output_path, style_set,
                         vline, hline, plot_configuration=None):
    from .plots.annotations import addCMSBits, labelAxis
    from .plots.common import PlotConfiguration
    from .plots.utils import saveFig
    from .style import Styler

    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    if len(group) != 1:
        raise RuntimeError("2D plot expects exactly one histogram per group")

    item, meta = group[0]
    h = item.histogram

    fig, ax = plt.subplots(layout="constrained")

    h.plot2d(ax=ax)

    # Add reference lines
    ax.axvline(x=vline, color="white", linestyle="--", linewidth=1.5)
    ax.axhline(y=hline, color="white", linestyle="--", linewidth=1.5)

    # Compute integrals for all four quadrants
    counts = h.values()
    x_centers = h.axes[0].centers
    y_centers = h.axes[1].centers

    # Exclude sentinel -1.0 bins
    valid_x = x_centers >= 0
    valid_y = y_centers >= 0
    valid_counts = counts[np.ix_(valid_x, valid_y)]
    vx = x_centers[valid_x]
    vy = y_centers[valid_y]

    total = np.sum(valid_counts)
    if total > 0:
        # Bottom-left: both < threshold (fully boosted)
        bl = np.sum(valid_counts[np.ix_(vx < vline, vy < hline)])
        # Bottom-right: x >= vline, y < hline
        br = np.sum(valid_counts[np.ix_(vx >= vline, vy < hline)])
        # Top-left: x < vline, y >= hline
        tl = np.sum(valid_counts[np.ix_(vx < vline, vy >= hline)])
        # Top-right: both >= threshold (fully resolved)
        tr = np.sum(valid_counts[np.ix_(vx >= vline, vy >= hline)])

        # Bottom-left corner — boosted
        ax.text(0.02, 0.18,
                f"Boosted\n{100*bl/total:.1f}%",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
        # Bottom-right corner
        ax.text(0.72, 0.18,
                f"Mixed\n{100*br/total:.1f}%",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
        # Top-left corner
        ax.text(0.02, 0.98,
                f"Mixed\n{100*tl/total:.1f}%",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
        # Top-right corner — resolved
        ax.text(0.72, 0.98,
                f"Resolved\n{100*tr/total:.1f}%",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    addCMSBits(
        ax,
        [meta],
        extra_text=f"{common_meta.get('pipeline', '')}",
        text_color="white",
        plot_configuration=pc,
    )

    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)

@define
class DRPlotWithLine2D(BasePostprocessor):
    output_name: str
    style_set: str | StyleSet = field(factory=StyleSet)
    vline: float = 0.8
    hline: float = 0.8

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            make2DDRPlotWithLine,
            group,
            common_meta,
            output_path,
            style_set=self.style_set,
            vline=self.vline,
            hline=self.hline,
            plot_configuration=pc,
        )
