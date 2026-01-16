import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from analyzer.postprocessing.style import Styler

from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .utils import saveFig


def plot2D(
    packaged_hist,
    output_path,
    style_set,
    normalize=False,
    plot_configuration=None,
    color_scale="linear",
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    fig, ax = plt.subplots(layout="constrained")
    styler.getStyle(packaged_hist.provenance.sector_parameters)
    h = packaged_hist.histogram

    if normalize:
        h = h / np.sum(h.values())
    if color_scale == "log":
        art = h.plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        art = h.plot2d(ax=ax)
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    sp = packaged_hist.provenance.sector_parameters
    addCMSBits(
        ax,
        [sp],
        extra_text=f"{sp.region_name}\n{packaged_hist.title}",
        text_color="white",
        plot_configuration=pc,
    )
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def getContour(HH, val):
    total = np.sum(HH)
    for i in range(round(np.max(HH))):
        if np.sum(HH[HH > i]) < (total * val):
            return i
    return None


def plot2DSigBkg(
    bkg_hist,
    sig_hist,
    output_path,
    style_set,
    normalize=False,
    plot_configuration=None,
    color_scale="linear",
    override_axis_labels=None,
):
    override_axis_labels = override_axis_labels or {}
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    fig, ax = plt.subplots(layout="constrained")
    styler.getStyle(bkg_hist.sector_parameters)
    h = bkg_hist.histogram

    if normalize:
        h = h / np.sum(h.values())
    if color_scale == "log":
        art = h.plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        art = h.plot2d(ax=ax)

    from scipy.ndimage import gaussian_filter

    sh = sig_hist.histogram

    HH, xe, ye = sh.to_numpy()
    HH = gaussian_filter(HH, 1.2)
    midpoints = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    grid = HH.transpose()
    h.sum().value

    sig_style = sig_hist.style or styler.getStyle(sig_hist.sector_parameters)

    ax.contour(
        *midpoints,
        grid,
        [getContour(HH, x) for x in (0.75, 0.5, 0.25)],
        linewidths=sig_style.line_width,
        colors=[sig_style.color],
    )

    labelAxis(ax, "y", h.axes, label=override_axis_labels.get("y"))
    labelAxis(ax, "x", h.axes, label=override_axis_labels.get("x"))

    proxy = [
        plt.Line2D(
            [0],
            [0],
            lw=sig_style.line_width or 2,
            color=sig_style.color,
            label=sig_hist.title,
        )
    ]

    sp = bkg_hist.sector_parameters
    ax.legend(
        handles=proxy,
        facecolor=pc.legend_fill_color,
        framealpha=pc.legend_fill_alpha,
        frameon=True,
    )

    addCMSBits(
        ax,
        [sp],
        extra_text=f"{sp.region_name}\n{bkg_hist.title}",
        text_color="white",
        plot_configuration=plot_configuration,
    )
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)
