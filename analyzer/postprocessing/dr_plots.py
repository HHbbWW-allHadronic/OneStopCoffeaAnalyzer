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


def makeDRPlotWithLine(group, common_meta, output_path, vline, vline_label):
    fig, ax = plt.subplots()

    all_counts = None
    for item, meta in group:
        h = item.histogram
        title = meta.get("title") or meta["dataset_title"]
        h.plot1d(ax=ax, label=title, flow="none")
        counts = h.values()
        all_counts = counts if all_counts is None else all_counts + counts

    # Add vertical dashed line
    ax.axvline(x=vline, color="black", linestyle="--", linewidth=1.5,
               label=f"{vline_label} = {vline}")

    # Compute integrals excluding sentinel -1.0 bin
    centers = h.axes[0].centers
    valid = centers >= 0  # exclude -1.0 sentinel bin
    valid_counts = all_counts[valid]
    valid_centers = centers[valid]
    total = np.sum(valid_counts)

    if total > 0:
        left = np.sum(valid_counts[valid_centers < vline])
        right = np.sum(valid_counts[valid_centers >= vline])
        ax.text(0.05, 0.95,
                f"< {vline}: {100*left/total:.1f}%\n> {vline}: {100*right/total:.1f}%",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    mplhep.cms.label("Preliminary", ax=ax)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
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
        yield ft.partial(
            makeDRPlotWithLine,
            group,
            common_meta,
            output_path,
            style_set=self.style_set,
            show_info=self.show_info,
            vline=self.vline,
            vline_label=self.vline_label,
        g
