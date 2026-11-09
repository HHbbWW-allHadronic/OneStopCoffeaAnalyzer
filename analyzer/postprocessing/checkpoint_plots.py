from __future__ import annotations
import csv
import functools as ft
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .style import StyleSet
from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    dotFormat,
)
from .processors import BasePostprocessor
from attrs import define, field


def parse_accuracy_from_pipeline(pipeline_name):
    """
    Pull the checkpoint's validation_average_jet_accuracy back out of a
    pipeline name like "withHLT_SPANet_v140_0p588" -> 0.588.
    """
    m = re.search(r"_(\d+)p(\d+)$", pipeline_name)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def makeModeOverlayPlot(group, common_meta, output_path, style_set,
                        normalize=False, plot_configuration=None):
    from .plots.annotations import addCMSBits, labelAxis
    from .plots.utils import saveFig
    from .plots.plots_1d import scaleYAxis
    from .plots.common import PlotConfiguration
    from .style import Styler

    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    fig, ax = plt.subplots()

    h = None
    mode_summary = []
    for item, meta in group:
        h = item.histogram
        label = meta.get("pipeline", meta.get("title", "?"))
        accuracy = parse_accuracy_from_pipeline(label)

        counts = h.values()
        centers = h.axes[0].centers
        valid = centers >= 0
        valid_counts = counts[valid]
        valid_centers = centers[valid]
        mode_value = valid_centers[np.argmax(valid_counts)] if valid_counts.sum() > 0 else float("nan")

        if valid_counts.sum() > 0:
            mean_value = np.average(valid_centers, weights=valid_counts)
            std_value = np.sqrt(np.average((valid_centers - mean_value) ** 2, weights=valid_counts))
        else:
            mean_value = float("nan")
            std_value = float("nan")

        mode_summary.append((label, accuracy, mode_value, mean_value, std_value))
        acc_str = f"acc={accuracy:.3f}, " if accuracy is not None else f"{label}, "
        legend_label = f"{acc_str}mode={mode_value:.1f}, mean={mean_value:.1f}, std={std_value:.1f}"

        style = styler.getStyle(meta)

        if normalize:
            total = h.sum().value
            h_display = h / total if total > 0 else h
        else:
            h_display = h
        h_display.plot1d(ax=ax, label=legend_label, flow="none", **style.get())

    labelAxis(ax, "y", h.axes, label=pc.y_label)
    labelAxis(ax, "x", h.axes, label=pc.x_label)
    addCMSBits(
        ax,
        [x.metadata for x in group],
        extra_text=f"{common_meta.get('name', '')}",
        plot_configuration=pc,
    )

    ax.legend(fontsize=11, loc="upper right")
    scaleYAxis(ax)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)

    csv_path = Path(str(output_path).rsplit(".", 1)[0] + "_mode_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "accuracy", "mode", "mean", "std"])
        for label, accuracy, mode_value, mean_value, std_value in mode_summary:
            writer.writerow([label, accuracy if accuracy is not None else "", mode_value, mean_value, std_value])
    print(f"Saved mode/accuracy summary: {csv_path}")

    return mode_summary


def makeModeVsAccuracyPlot(group, common_meta, output_path, plot_configuration=None):
    from .plots.annotations import addCMSBits
    from .plots.common import PlotConfiguration
    from .plots.utils import saveFig

    pc = plot_configuration or PlotConfiguration()

    pipelines, accuracies, modes = [], [], []
    for item, meta in group:
        label = meta.get("pipeline", meta.get("title", "?"))
        accuracy = parse_accuracy_from_pipeline(label)
        if accuracy is None:
            print(f"  WARNING: could not parse accuracy from pipeline name '{label}', skipping")
            continue

        h = item.histogram
        counts = h.values()
        centers = h.axes[0].centers
        valid = centers >= 0
        valid_counts = counts[valid]
        valid_centers = centers[valid]
        mode_value = valid_centers[np.argmax(valid_counts)] if valid_counts.sum() > 0 else float("nan")

        pipelines.append(label)
        accuracies.append(accuracy)
        modes.append(mode_value)

    accuracies = np.array(accuracies)
    modes = np.array(modes)
    n = len(accuracies)

    if n < 2:
        print(f"  WARNING: only {n} valid (accuracy, mode) point(s) -- need at least 2 to fit "
              f"a line, skipping ModeVsAccuracyPlot output")
        return None

    slope, intercept = np.polyfit(accuracies, modes, 1)
    fit_y = slope * accuracies + intercept
    ss_res = np.sum((modes - fit_y) ** 2)
    ss_tot = np.sum((modes - modes.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = np.corrcoef(accuracies, modes)[0, 1] if n > 1 else float("nan")

    fig, ax = plt.subplots()
    ax.scatter(accuracies, modes, color="#2196F3", s=60, zorder=3, label="Checkpoints")

    order = np.argsort(accuracies)
    sorted_pipelines = [pipelines[i] for i in order]
    sorted_accuracies = accuracies[order]
    sorted_modes = modes[order]

    for rank, (x, y) in enumerate(zip(sorted_accuracies, sorted_modes), start=1):
        offset = (6, 6) if rank % 2 else (6, -14)
        ax.annotate(str(rank), (x, y), textcoords="offset points", xytext=offset,
                    fontsize=9, fontweight="bold")

    key_lines = "\n".join(
        f"{rank}: {p} (acc={x:.3f}, mode={y:.1f})"
        for rank, (p, x, y) in enumerate(zip(sorted_pipelines, sorted_accuracies, sorted_modes), start=1)
    )
    ax.text(0.98, 0.98, key_lines, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_xlabel("validation_average_jet_accuracy", fontsize=12)
    ax.set_ylabel("Mode mass value [GeV]", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    addCMSBits(
        ax,
        [x.metadata for x in group],
        extra_text=f"{common_meta.get('name', '')}",
        plot_configuration=pc,
    )

    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)
    print(f"slope={slope:.3f}  intercept={intercept:.3f}  r={r:.3f}  R^2={r_squared:.3f}  n={n}")


@define
class ModeVsAccuracyPlot(BasePostprocessor):
    """
    Builds the mode-mass-vs-validation-accuracy scatter+fit plot directly
    from the group (no CSV/external script needed) -- reuses the exact
    same grouping mechanism as ModeOverlayPlot (structure.group WITHOUT
    `pipeline`, so all checkpoints for a given histogram name land in one
    group) and the same parse_accuracy_from_pipeline() regex.

    NOTE: same small-n caveat as ModeOverlayPlot's mode estimate -- with
    only a handful of checkpoints, R^2/slope can shift substantially from
    a single point. The plot annotates this directly rather than implying
    more confidence than the sample size supports.
    """
    output_name: str

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeModeVsAccuracyPlot,
            group,
            common_meta,
            output_path,
            plot_configuration=pc,
        )


@define
class ModeOverlayPlot(BasePostprocessor):
    """
    Overlay several histograms of the SAME name (e.g. from different
    checkpoints/pipelines run against the same dataset) on one plot, with
    each line's mode (tallest-bin value) annotated in its legend label.

    NOTE: mode-from-tallest-bin is a simple estimator, not a fitted peak --
    with thin statistics it can be noisy bin-to-bin. Treat trends across
    checkpoints as indicative rather than exact; a windowed Gaussian/parabola
    fit around the tallest bin would be the natural upgrade if this proves
    too noisy to read.

    """
    output_name: str
    normalize: bool = False

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeModeOverlayPlot,
            group,
            common_meta,
            output_path,
            style_set=self.style_set,
            normalize=self.normalize,
            plot_configuration=pc,
        )
