from __future__ import annotations
import functools as ft
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import hist
from .style import StyleSet
from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    dotFormat,
)
from .processors import BasePostprocessor
from attrs import define, field


def makeMassWindowProjection(group, common_meta, output_path, style_set, show_info,
                              cut_axis, cut_low, cut_high, normalize, scale,
                              plot_configuration=None):
    from .plots.annotations import addCMSBits, labelAxis
    from .plots.utils import addLegend, saveFig, scaleYAxis
    from .plots.common import PlotConfiguration
    from .style import Styler

    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    fig, ax = plt.subplots()

    projected_axes = None
    for item, meta in group:
        h2d = item.histogram
        slicer = [slice(None), slice(None)]
        slicer[cut_axis] = slice(hist.loc(cut_low), hist.loc(cut_high), sum)
        h = h2d[tuple(slicer)]
        if projected_axes is None:
            projected_axes = h.axes

        title = meta.get("title") or meta["dataset_title"]
        if show_info:
            integral = h.sum().value
            counts = h.values()
            centers = h.axes[0].centers
            total_counts = counts.sum()
            if total_counts > 0:
                mean = np.average(centers, weights=counts)
                std = np.sqrt(np.average((centers - mean) ** 2, weights=counts))
                title = f"{title}, Int.={integral:.1f}\nmean={mean:.3f}, std={std:.3f}"
            else:
                title = f"{title}, Int.={integral:.1f} (no events in window)"
        style = styler.getStyle(meta)
        h.plot1d(ax=ax, label=title, density=normalize, yerr=style.yerr, flow="none", **style.get())

    labelAxis(ax, "y", projected_axes, label=pc.y_label)
    labelAxis(ax, "x", projected_axes, label=pc.x_label)
    addCMSBits(
        ax,
        [x.metadata for x in group],
        extra_text=f"{common_meta.get('pipeline', '')}",
        plot_configuration=pc,
    )
    addLegend(ax, pc)
    ax.set_yscale(scale)
    scaleYAxis(ax)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


@define
class MassWindowProjection(BasePostprocessor):
    """
    Cuts a 2D histogram along cut_axis to [cut_low, cut_high), summing
    that axis away, and plots the resulting 1D projection over the
    remaining axis -- e.g. cut MWW to [50, 200], plot the Mbb
    distribution for events landing in that window.

    Each item in `group` is drawn as its own overlaid 1D line, same
    pattern as DRPlotWithLine -- meant for a direct shape comparison
    across datasets (e.g. signal vs. a background), not a stacked
    yield estimate. For that overlay to actually happen, `group`'s
    structure.group in the YAML must NOT include dataset_name --
    grouping only by {pipeline, name} (or similar) is what puts
    multiple datasets into the same group so they land on one plot
    together, rather than producing one plot per dataset.

    cut_axis : int
        Which of the 2D histogram's two axes (0 or 1) to cut on and
        sum away. The OTHER axis is what gets plotted. Matches
        whichever axis position the input_cols the 2D histogram was
        built from actually used -- check the analyzer-side YAML's
        SimpleHistogram input_cols order for the histogram being fed
        in here to get this right.
    normalize : bool
        Defaults to True (density-normalized). For a raw-yield
        comparison instead (e.g. signal vs. QCD where the actual,
        weighted event counts matter, not just relative shape), set
        this False and use scale="log" instead -- density normalization
        and log scale solve the same "wildly different yields" problem
        in different ways; they're independent options here, not tied
        together, so either or both can be set as needed.
    scale : "log" or "linear"
        Y-axis scale. Defaults to "linear", matching Histogram1D/
        Histogram2D's own scale field elsewhere in this codebase.
    """
    output_name: str
    cut_axis: int
    cut_low: float
    cut_high: float
    style_set: str | StyleSet = field(factory=StyleSet)
    show_info: bool = True
    normalize: bool = True
    scale: Literal["log", "linear"] = "linear"

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeMassWindowProjection,
            group,
            common_meta,
            output_path,
            style_set=self.style_set,
            show_info=self.show_info,
            cut_axis=self.cut_axis,
            cut_low=self.cut_low,
            cut_high=self.cut_high,
            normalize=self.normalize,
            scale=self.scale,
            plot_configuration=pc,
        )
