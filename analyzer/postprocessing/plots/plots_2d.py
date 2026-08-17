import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from analyzer.postprocessing.style import Styler

from analyzer.utils.structure_tools import commonDict
from .annotations import labelAxis
from .common import PlotConfiguration
from .utils import saveFigVariants
import mplhep

import functools as ft
import operator as op
import hist
from .plots_1d import computeRatio, computeSignificance


def plot2D(
    histogram,
    common_meta,
    output_path,
    style_set,
    normalize=False,
    plot_configuration=None,
    color_scale="linear",
    vline=None,
    hline=None,
    cbar_title="Events",
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    fig, ax = plt.subplots(layout="constrained")
    item, meta = histogram
    h = item.histogram
    style = styler.getStyle(meta)
    if normalize:
        h = h / np.sum(h.values())
    if color_scale == "log":
        objs = mplhep.hist2dplot(h, norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        objs = mplhep.hist2dplot(h, ax=ax)
    cbar = objs.cbar
    if cbar_title and cbar is not None:
        cbar.set_label(cbar_title)

    # Add optional reference lines
    if vline is not None:
        ax.axvline(x=vline, color="white", linestyle="--", linewidth=1.5)
    if hline is not None:
        ax.axhline(y=hline, color="white", linestyle="--", linewidth=1.5)

    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    saveFigVariants(
        fig,
        ax,
        output_path,
        [meta],
        plot_configuration=pc,
        metadata=common_meta,
        extra_text=f"{common_meta['pipeline']}",
        text_color=pc.cms_text_color or "white",
    )
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
        h.plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        h.plot2d(ax=ax)

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

    common_meta = commonDict([bkg_hist.metadata, sig_hist.metadata], key=lambda x: x)
    saveFigVariants(
        fig,
        ax,
        output_path,
        [sp],
        plot_configuration=pc,
        metadata=common_meta,
        extra_text=f"{sp.region_name}\n{bkg_hist.title}",
        text_color=plot_configuration.cms_text_color or "white",
    )
    plt.close(fig)


RATIO_FUNCS = {
    "poisson": computeRatio,
    "poisson-ratio": computeRatio,
    "efficiency": computeRatio,
    "significance": computeSignificance,
}


def getRatioFunc(ratio_type):
    try:
        return RATIO_FUNCS[ratio_type]
    except KeyError:
        raise ValueError(
            f"Unknown ratio_type '{ratio_type}', expected one of {sorted(RATIO_FUNCS)}"
        ) from None


# Presentation defaults per mode. A ratio centred on 1 wants a diverging map;
# an efficiency wants a fixed [0,1] sequential map.
RATIO_MODE_DEFAULTS = {
    "poisson": dict(label="Ratio", cmap="coolwarm", center=1.0, z_range=None),
    "poisson-ratio": dict(label="Ratio", cmap="coolwarm", center=1.0, z_range=None),
    "efficiency": dict(
        label="Efficiency", cmap="viridis", center=None, z_range=(0.0, 1.0)
    ),
    "significance": dict(
        label=r"Significance", cmap="viridis", center=None, z_range=None
    ),
}


def makeRatioHist(
    num_h, den_h, ratio_type="poisson", normalize=False, mask_zeros=False
):
    if len(num_h.axes) != 2:
        raise ValueError(f"Expected a 2D histogram, got {len(num_h.axes)} axes")
    if num_h.shape != den_h.shape:
        raise ValueError(f"Shape mismatch: {num_h.shape} vs {den_h.shape}")

    n, d = num_h.values(), den_h.values()
    values, unc = getRatioFunc(ratio_type)(
        n, d, normalize=normalize, ratio_type=ratio_type
    )
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = np.nan

    # computeRatio blanks exact zeros, which is wrong for an efficiency map:
    # 0/N is a real measurement, not an undefined bin.
    if not mask_zeros:
        values[(n == 0) & (d > 0)] = 0.0

    rh = hist.Hist(*num_h.axes, storage=hist.storage.Double())
    rh.view(flow=False)[...] = values
    return rh, unc


def makeRatioNorm(values, color_scale, center, z_range):
    finite = values[np.isfinite(values)]
    if z_range is not None:
        vmin, vmax = z_range
    elif finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0

    if color_scale == "log":
        positive = finite[finite > 0]
        lo = float(positive.min()) if positive.size else 1e-3
        return matplotlib.colors.LogNorm(vmin=max(vmin, lo), vmax=max(vmax, lo * 10))
    if center is not None:
        # TwoSlopeNorm requires vmin < vcenter < vmax strictly
        vmin, vmax = min(vmin, center - 1e-9), max(vmax, center + 1e-9)
        return matplotlib.colors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


def quadratureSum(rh, den_h=None, min_den=None, top_n=5):
    """TEMPORARY DIAGNOSTIC -- delete along with its call site in plotRatio2D.

    Combines per-bin significances: Z_tot = sqrt(sum_i Z_i^2).
    Only meaningful for ratio_type="significance".
    """
    z = np.asarray(rh.values(), dtype=float)
    finite = np.isfinite(z)

    if min_den is not None:
        if den_h is None:
            raise ValueError("min_den requires den_h")
        finite &= den_h.values() >= min_den

    z2 = np.where(finite, z, 0.0) ** 2
    total_sq = z2.sum()
    total = float(np.sqrt(total_sq))

    print(f"Z_total = {total:.4f}  ({int(finite.sum())}/{z.size} bins used)")
    xc, yc = rh.axes[0].centers, rh.axes[1].centers
    print(f"  top {top_n} contributing bins:")
    for idx in np.argsort(z2, axis=None)[::-1][:top_n]:
        i, j = np.unravel_index(idx, z2.shape)
        frac = z2[i, j] / total_sq if total_sq else 0.0
        print(
            f"    (x={xc[i]:.4g}, y={yc[j]:.4g})  Z={z[i, j]:.4f}  "
            f"{100 * frac:5.1f}% of Z_total^2"
        )
    return total


def plotRatio2D(
    denominator,  # list of (item, meta) -- summed
    numerators,  # list of (item, meta) -- must be length 1
    common_meta,
    output_path,
    style_set=None,  # unused for now
    ratio_type="poisson",
    normalize=False,
    plot_configuration=None,
    color_scale="linear",
    z_range=None,
    center="auto",  # "auto" -> mode default, None -> no diverging norm
    cmap=None,
    cbar_title=None,
    mask_zeros=False,
    vline=None,
    hline=None,
):
    pc = plot_configuration or PlotConfiguration()
    defaults = RATIO_MODE_DEFAULTS.get(ratio_type, {})

    if len(numerators) != 1:
        raise RuntimeError(
            f"plotRatio2D takes exactly 1 numerator, got {len(numerators)}"
        )
    if not denominator:
        raise RuntimeError("plotRatio2D needs at least 1 denominator")

    num_item, num_meta = numerators[0]
    num_h = num_item.histogram
    den_h = ft.reduce(op.add, (item.histogram for item, _ in denominator))

    rh, unc = makeRatioHist(
        num_h, den_h, ratio_type=ratio_type, normalize=normalize, mask_zeros=mask_zeros
    )
    print(output_path)
    quadratureSum(
        rh, den_h=den_h, min_den=1.0
    )  # TEMPORARY DIAGNOSTIC -- delete this line
    if z_range is None:
        z_range = defaults.get("z_range")
    if center == "auto":
        center = defaults.get("center")
    if color_scale == "log":
        center = None
    norm = makeRatioNorm(rh.values(), color_scale, center, z_range)

    cmap_obj = matplotlib.colormaps[cmap or defaults.get("cmap", "viridis")].copy()
    cmap_obj.set_bad(color="none")  # undefined bins render blank

    fig, ax = plt.subplots(layout="constrained")
    objs = mplhep.hist2dplot(rh, ax=ax, cmap=cmap_obj, norm=norm)
    if objs.cbar is not None:
        objs.cbar.set_label(cbar_title or defaults.get("label", "Ratio"))
        fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        objs.cbar.ax.yaxis.set_major_formatter(fmt)

    if vline is not None:
        ax.axvline(x=vline, color="white", linestyle="--", linewidth=1.5)
    if hline is not None:
        ax.axhline(y=hline, color="white", linestyle="--", linewidth=1.5)

    labelAxis(ax, "y", rh.axes)
    labelAxis(ax, "x", rh.axes)

    all_meta = [num_meta] + [m for _, m in denominator]
    saveFigVariants(
        fig,
        ax,
        output_path,
        all_meta,
        plot_configuration=pc,
        metadata=common_meta,
        extra_text=f"{common_meta.get('pipeline', '')}",
        text_color=pc.cms_text_color or "white",
    )
    plt.close(fig)
