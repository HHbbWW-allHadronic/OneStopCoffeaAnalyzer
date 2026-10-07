"""
Detection-score postprocessors for OSCA: the detection-bin mass-plane / S/sqrt(B) study and the
mass-after-detection-cut projections that batch_trace.py makes, but on OSCA's own weighted histograms.

Both classes take ONE 3D histogram per sample, with axes
    (reconstructed Hbb mass, reconstructed on-shell-W mass, detection score)
in whatever axis ORDER the analyzer-side SimpleHistogram used -- set h_axis / w_axis / det_axis to match
(defaults 0, 1, 2). The group must hold the signal histogram(s) AND the QCD histogram(s) of the same
pipeline/histogram, so group on {pipeline, name} and NOT on dataset_name (same rule as MassWindowProjection).
Signal and QCD items are told apart by regex on meta["dataset_name"]. Several QCD datasets (pT-hat bins)
or several signal datasets are summed.

Weights: histograms are used as OSCA made them (cross-section x lumi / N_gen weights already applied), so the
default scales are 1. signal_scale / bkg_scale are extra constant factors (e.g. a fraction of the generated
events). Uncertainties come from the histogram variances. The "minimum QCD events" guard uses the EFFECTIVE
event count n_eff = (sum w)^2 / (sum w^2), which equals the raw count for unit weights and is what limits a
weighted estimate; cells/windows below min_b_events are masked, not shown as a huge S/sqrt(B).

Everything is computed from h.values()/h.variances() WITHOUT under/overflow bins.

Detection ranges / cuts are snapped to the detection axis' bin edges (a warning is printed when an
edge you asked for is not an edge of the axis); use a detection axis with 100 bins on [0, 1] so the default
edges are exact.
"""
from __future__ import annotations

import csv
import functools as ft
import re
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import hist
from attrs import define, field

from .style import StyleSet
from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    dotFormat,
)
from .processors import BasePostprocessor

DEFAULT_DET_BIN_EDGES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
_IMG_EXTS = (".png", ".pdf", ".jpg", ".jpeg", ".svg")


# ----------------------------------------------------------------------------------------------
# pure helpers (no OSCA objects) -- these hold all the numbers
# ----------------------------------------------------------------------------------------------
def split_ext(output_path):
    """(path without a trailing image extension, that extension or ''). Your output_name values end in
    '.png', so derived files (summary, csv) have to be built around the extension, not after it."""
    p = str(output_path)
    for e in _IMG_EXTS:
        if p.lower().endswith(e):
            return p[:-len(e)], p[-len(e):]
    return p, ""


def split_group(group, signal_regex, background_regex):
    """Split (item, meta) pairs into signal / background by regex on meta['dataset_name']."""
    sig, bkg, other = [], [], []
    for x in group:
        item, meta = x
        name = str(meta.get("dataset_name", ""))
        is_s = re.search(signal_regex, name) is not None
        is_b = re.search(background_regex, name) is not None
        if is_s and is_b:
            raise RuntimeError(
                f"dataset_name {name!r} matches BOTH signal_regex {signal_regex!r} and background_regex "
                f"{background_regex!r}; make the patterns disjoint."
            )
        if is_s:
            sig.append(x)
        elif is_b:
            bkg.append(x)
        else:
            other.append(name)
    return sig, bkg, other


def sum_histograms(pairs):
    total = None
    for x in pairs:
        h = x[0].histogram
        total = h.copy() if total is None else total + h
    return total


def hist_arrays(h, h_axis, w_axis, det_axis):
    """3D hist -> (values, variances) transposed to (H, W, detection) plus the three edge arrays.
    No flow bins. Double-storage histograms (no variances) are treated as Poisson counts."""
    if h.ndim != 3:
        raise RuntimeError(f"expected a 3D histogram (m_H, m_W, detection), got {h.ndim}D with axes "
                           f"{[a.name for a in h.axes]}")
    perm = (h_axis, w_axis, det_axis)
    if sorted(perm) != [0, 1, 2]:
        raise RuntimeError(f"h_axis, w_axis, det_axis must be a permutation of 0,1,2, got {perm}")
    v = np.asarray(h.values(), dtype=float)
    var = h.variances()
    var = v.copy() if var is None else np.asarray(var, dtype=float)
    return (np.transpose(v, perm), np.transpose(var, perm),
            [np.asarray(h.axes[a].edges, dtype=float) for a in perm])


def snap_edge(edges, x, tol=1e-6):
    """Index of the axis edge nearest x, and whether x was (numerically) an edge."""
    i = int(np.argmin(np.abs(edges - x)))
    return i, bool(abs(edges[i] - x) <= tol * max(float(edges[-1] - edges[0]), 1.0))


def _z(S, varS, B, varB, n_eff_B, min_b):
    """S/sqrt(B) with first-order error propagation; (nan, nan) if B is not usable."""
    if not (B > 0) or n_eff_B < min_b:
        return np.nan, np.nan
    z = S / np.sqrt(B)
    err = np.sqrt(max(varS / B + (S ** 2) * varB / (4.0 * B ** 3), 0.0))
    return float(z), float(err)


def _neff(sumw, sumw2):
    return float(sumw ** 2 / sumw2) if sumw2 > 0 else 0.0


def region_stats(Vs, Ws, Vb, Wb, det_slice, signal_scale, bkg_scale, win_mask, min_b):
    """Statistics for events whose detection bin lies in det_slice.
    Vs/Ws, Vb/Wb : (nH, nW, nDet) weight sums / variances for signal / background (unscaled)."""
    s_u = Vs[:, :, det_slice].sum(axis=2)
    sv_u = Ws[:, :, det_slice].sum(axis=2)
    b_u = Vb[:, :, det_slice].sum(axis=2)
    bv_u = Wb[:, :, det_slice].sum(axis=2)
    S2, varS2 = s_u * signal_scale, sv_u * signal_scale ** 2
    B2, varB2 = b_u * bkg_scale, bv_u * bkg_scale ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        neff_cell = np.where(bv_u > 0, b_u ** 2 / bv_u, 0.0)
        valid = (neff_cell >= min_b) & (B2 > 0)
        zmap = np.where(valid, S2 / np.sqrt(np.where(B2 > 0, B2, np.nan)), np.nan)
        zquad = float(np.sqrt(np.nansum(np.where(valid, S2 ** 2 / np.where(B2 > 0, B2, np.nan), 0.0)))) \
            if valid.any() else np.nan

    S, varS = float(S2.sum()), float(varS2.sum())
    B, varB = float(B2.sum()), float(varB2.sum())
    z_all, e_all = _z(S, varS, B, varB, _neff(b_u.sum(), bv_u.sum()), min_b)
    Sw, varSw = float(S2[win_mask].sum()), float(varS2[win_mask].sum())
    Bw, varBw = float(B2[win_mask].sum()), float(varB2[win_mask].sum())
    z_win, e_win = _z(Sw, varSw, Bw, varBw, _neff(b_u[win_mask].sum(), bv_u[win_mask].sum()), min_b)
    return {
        "S2": S2, "B2": B2, "zmap": zmap,
        "S": S, "B": B, "neff_B": _neff(b_u.sum(), bv_u.sum()),
        "z_all": z_all, "z_all_err": e_all,
        "S_win": Sw, "B_win": Bw, "neff_B_win": _neff(b_u[win_mask].sum(), bv_u[win_mask].sum()),
        "z_window": z_win, "z_window_err": e_win,
        "z_quad": zquad, "n_cells_used": int(valid.sum()),
    }


def window_mask(h_edges, w_edges, higgs_mass, w_mass, h_window, w_window):
    """(nH, nW) bool: cell centres inside the |m_H - higgs_mass| <= h_window, |m_W - w_mass| <= w_window box."""
    hc = 0.5 * (h_edges[:-1] + h_edges[1:])
    wc = 0.5 * (w_edges[:-1] + w_edges[1:])
    return (np.abs(hc[:, None] - higgs_mass) <= h_window) & (np.abs(wc[None, :] - w_mass) <= w_window)


def make_1d_hist(axis, values, variances):
    h1 = hist.Hist(axis, storage=hist.storage.Weight())
    view = h1.view(flow=False)
    view["value"] = values
    view["variance"] = variances
    return h1


# ----------------------------------------------------------------------------------------------
# plot 1: detection-bin 2D mass planes + S/sqrt(B) vs cumulative detection cut
# ----------------------------------------------------------------------------------------------
def makeDetectionBinned2D(group, common_meta, output_path, signal_regex, background_regex,
                          h_axis, w_axis, det_axis, det_bin_edges, signal_scale, bkg_scale,
                          min_b_events, higgs_mass, w_mass, h_window, w_window,
                          plot_configuration=None):
    from .plots.annotations import addCMSBits
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    sig, bkg, other = split_group(group, signal_regex, background_regex)
    if other:
        print(f"[DetectionBinned2D] ignoring datasets matching neither regex: {sorted(set(other))}")
    if not sig or not bkg:
        raise RuntimeError(
            f"need signal AND background in one group (found {len(sig)} signal, {len(bkg)} background). Group on "
            f"{{pipeline, name}} without dataset_name so signal and QCD land together; check signal_regex "
            f"{signal_regex!r} / background_regex {background_regex!r} against meta['dataset_name'].")
    Vs, Ws, (h_edges, w_edges, d_edges) = hist_arrays(sum_histograms(sig), h_axis, w_axis, det_axis)
    Vb, Wb, e2 = hist_arrays(sum_histograms(bkg), h_axis, w_axis, det_axis)
    if not all(np.array_equal(a, b) for a, b in zip((h_edges, w_edges, d_edges), e2)):
        raise RuntimeError("signal and background histograms have different axes")
    n_det = len(d_edges) - 1
    win = window_mask(h_edges, w_edges, higgs_mass, w_mass, h_window, w_window)

    # detection ranges (grid rows), snapped to axis edges
    wanted = sorted(set(det_bin_edges))
    snapped = []
    for x in wanted:
        i, exact = snap_edge(d_edges, x)
        if not exact:
            print(f"[DetectionBinned2D] detection edge {x:g} is not an edge of the detection axis; using {d_edges[i]:g}")
        snapped.append(i)
    snapped = sorted(set(snapped))
    ranges = list(zip(snapped[:-1], snapped[1:]))
    results = [((d_edges[a], d_edges[b]), region_stats(Vs, Ws, Vb, Wb, slice(a, b), signal_scale, bkg_scale, win,
                                                        min_b_events)) for a, b in ranges]
    # cumulative scan at EVERY detection bin edge: keep detection >= cut
    cum = [(d_edges[i], region_stats(Vs, Ws, Vb, Wb, slice(i, n_det), signal_scale, bkg_scale, win, min_b_events))
           for i in range(n_det)]

    fmt = lambda v, p=2: "  -  " if (v is None or not np.isfinite(v)) else f"{v:.{p}f}"
    title_extra = f"{common_meta.get('pipeline', '')}"
    print(f"\n=== Detection-bin mass plane: signal {len(sig)} dataset(s), background {len(bkg)}; "
          f"scales signal x{signal_scale:g}, background x{bkg_scale:g}; window H {higgs_mass:g}+/-{h_window:g}, "
          f"W {w_mass:g}+/-{w_window:g}; min n_eff(B) {min_b_events:g} ===")
    hdr = f"  {'detection':>13s} {'S':>10s} {'B':>10s} {'nEffB':>8s} {'S/sqrtB all':>13s} {'in window':>16s} {'quad.':>8s}"
    print(hdr)
    for (lo, hi), s in results:
        print(f"  {f'[{lo:.2f},{hi:.2f}' + (']' if hi == d_edges[-1] else ')'):>13s} {s['S']:10.3g} {s['B']:10.3g} "
              f"{s['neff_B']:8.1f} {fmt(s['z_all']):>13s} "
              f"{(fmt(s['z_window']) + ' +/- ' + fmt(s['z_window_err'])):>16s} {fmt(s['z_quad']):>8s}")
    best = max(((c, s) for c, s in cum if np.isfinite(s["z_window"])), key=lambda cs: cs[1]["z_window"], default=None)
    if best:
        print(f"  best cumulative cut by in-window S/sqrt(B): detection >= {best[0]:.3f} "
              f"(z = {best[1]['z_window']:.2f} +/- {best[1]['z_window_err']:.2f}, "
              f"n_eff(B in window) = {best[1]['neff_B_win']:.1f})")

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    cols = ["mode", "lo", "hi", "S", "B", "neff_B", "z_all", "z_all_err", "S_win", "B_win", "neff_B_win",
            "z_window", "z_window_err", "z_quad", "n_cells_used"]
    with open(f"{stem}_significance.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for (lo, hi), s in results:
            wr.writerow({"mode": "range", "lo": lo, "hi": hi, **{k: s[k] for k in cols[3:]}})
        for c, s in cum:
            wr.writerow({"mode": "cumulative", "lo": c, "hi": d_edges[-1], **{k: s[k] for k in cols[3:]}})

    # ---- grid: rows = detection ranges; signal | background | S/sqrt(B) per cell ----
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.4 * n), squeeze=False)
    for i, ((lo, hi), s) in enumerate(results):
        for j, (ttl, arr, cmap) in enumerate((("signal", s["S2"], "viridis"), ("QCD", s["B2"], "magma"),
                                              ("S/sqrt(B) per cell", s["zmap"], "plasma"))):
            ax = axes[i][j]
            data = np.ma.masked_invalid(arr).T
            if j < 2:
                data = np.ma.masked_where(data <= 0, data)
            mesh = ax.pcolormesh(h_edges, w_edges, data, cmap=cmap, shading="flat")
            fig.colorbar(mesh, ax=ax)
            ax.set_title(f"detection [{lo:.2f}, {hi:.2f}{']' if i == n - 1 else ')'} -- {ttl}"
                         + (f"  (S={s['S']:.3g})" if j == 0 else f"  (B={s['B']:.3g}, n_eff={s['neff_B']:.0f})" if j == 1 else ""),
                         fontsize=9)
            if j == 0:
                ax.add_patch(plt.Rectangle((higgs_mass - h_window, w_mass - w_window), 2 * h_window, 2 * w_window,
                                           fill=False, edgecolor="w", linestyle="--", linewidth=1))
            ax.set_xlabel("m(H1 pick) [GeV]")
            ax.set_ylabel("m(W1 pick) [GeV]")
    fig.suptitle(f"{title_extra}  Reconstructed mass plane by detection-score range. Dashed box = window; "
                 f"S/sqrt(B) cells need n_eff(B) >= {min_b_events:g}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    saveFig(fig, str(output_path), extension=pc.image_type)
    plt.close(fig)

    # ---- summary: significance and efficiencies vs cumulative detection cut ----
    fig, (ax_z, ax_e) = plt.subplots(1, 2, figsize=(14, 5))
    cx = np.array([c for c, _ in cum])
    zw = np.array([s["z_window"] for _, s in cum])
    zw_e = np.array([s["z_window_err"] for _, s in cum])
    ax_z.plot(cx, [s["z_all"] for _, s in cum], color="#7f8c8d", label="all events")
    ax_z.plot(cx, zw, color="#c0392b", label="in the H/W window")
    ok = np.isfinite(zw) & np.isfinite(zw_e)
    ax_z.fill_between(cx[ok], (zw - zw_e)[ok], (zw + zw_e)[ok], color="#c0392b", alpha=0.2, linewidth=0)
    ax_z.plot(cx, [s["z_quad"] for _, s in cum], color="#8e44ad", label="quadrature over unmasked cells")
    if best:
        ax_z.axvline(best[0], color="k", linestyle=":", linewidth=1)
    ax_z.set_xlabel("detection-score cut (keep >= cut)")
    ax_z.set_ylabel("S/sqrt(B)")
    ax_z.set_title("significance vs detection cut")
    ax_z.legend(loc="upper right")
    s0, b0 = cum[0][1]["S"], cum[0][1]["B"]
    ax_e.plot(cx, [s["S"] / s0 if s0 else np.nan for _, s in cum], color="#2a9959", label="signal kept")
    ax_e.plot(cx, [s["B"] / b0 if b0 else np.nan for _, s in cum], color="#c0392b", label="QCD kept")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("detection-score cut (keep >= cut)")
    ax_e.set_ylabel("fraction kept (weighted)")
    ax_e.set_title("efficiencies")
    ax_e.legend()
    addCMSBits(ax_z, [x.metadata for x in group], extra_text=title_extra, plot_configuration=pc)
    saveFig(fig, f"{stem}_summary{ext}", extension=pc.image_type)
    plt.close(fig)


@define
class DetectionBinned2D(BasePostprocessor):
    """
    Detection-score-binned Hbb x W mass planes, signal and QCD side by side with S/sqrt(B) per cell, plus
    S/sqrt(B) and the kept fractions against a cumulative detection cut. See the module docstring for the
    required 3D input histogram, grouping and weighting rules.

    Writes <output_name> (the grid), <output_name minus extension>_summary.<ext> and
    <output_name minus extension>_significance.csv.

    signal_regex, background_regex : regexes searched in meta["dataset_name"]; must be disjoint.
    h_axis, w_axis, det_axis       : positions of m_Hbb, m_W and the detection score in the 3D histogram.
    det_bin_edges                  : detection ranges (grid rows).
    signal_scale, bkg_scale        : extra constant factors on the weighted yields (default 1).
    min_b_events                   : minimum effective QCD events for a cell/window/total to be used.
    higgs_mass, w_mass, h_window, w_window : the box for the in-window significance.
    """
    output_name: str
    signal_regex: str
    background_regex: str
    h_axis: int = 0
    w_axis: int = 1
    det_axis: int = 2
    det_bin_edges: List[float] = field(factory=lambda: list(DEFAULT_DET_BIN_EDGES))
    signal_scale: float = 1.0
    bkg_scale: float = 1.0
    min_b_events: float = 5.0
    higgs_mass: float = 125.0
    w_mass: float = 80.4
    h_window: float = 20.0
    w_window: float = 15.0

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeDetectionBinned2D, group, common_meta, output_path,
            signal_regex=self.signal_regex, background_regex=self.background_regex,
            h_axis=self.h_axis, w_axis=self.w_axis, det_axis=self.det_axis,
            det_bin_edges=self.det_bin_edges, signal_scale=self.signal_scale, bkg_scale=self.bkg_scale,
            min_b_events=self.min_b_events, higgs_mass=self.higgs_mass, w_mass=self.w_mass,
            h_window=self.h_window, w_window=self.w_window, plot_configuration=pc,
        )


# ----------------------------------------------------------------------------------------------
# plot 2: mass projection before / after a detection cut, signal vs QCD
# ----------------------------------------------------------------------------------------------
def makeDetectionCutMassProjection(group, common_meta, output_path, style_set, signal_regex, background_regex,
                                   h_axis, w_axis, det_axis, mass_axis, det_cut, normalize, scale,
                                   plot_configuration=None):
    from .plots.annotations import addCMSBits, labelAxis
    from .plots.utils import addLegend, saveFig, scaleYAxis
    from .plots.common import PlotConfiguration
    from .style import Styler

    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    sig, bkg, other = split_group(group, signal_regex, background_regex)
    if other:
        print(f"[DetectionCutMassProjection] ignoring datasets matching neither regex: {sorted(set(other))}")
    if not sig or not bkg:
        raise RuntimeError(f"need signal AND background in one group (found {len(sig)} signal, {len(bkg)} background).")
    if mass_axis not in ("h", "w"):
        raise RuntimeError("mass_axis must be 'h' (m_Hbb) or 'w' (m_W)")
    fig, ax = plt.subplots()
    last_h1 = None
    cut_idx = None
    for label, pairs in (("signal", sig), ("QCD", bkg)):
        V, W, (h_edges, w_edges, d_edges) = hist_arrays(sum_histograms(pairs), h_axis, w_axis, det_axis)
        if cut_idx is None:
            cut_idx, exact = snap_edge(d_edges, det_cut)
            if not exact:
                print(f"[DetectionCutMassProjection] cut {det_cut:g} is not a detection-axis edge; using {d_edges[cut_idx]:g}")
        src = pairs[0][0].histogram
        mass_ax = src.axes[h_axis if mass_axis == "h" else w_axis]
        sum_axis = 1 if mass_axis == "h" else 0     # sum the OTHER mass axis away
        for sel, tag, dashed in ((slice(0, None), "no cut", True), (slice(cut_idx, None), f"detection >= {d_edges[cut_idx]:g}", False)):
            v = V[:, :, sel].sum(axis=2).sum(axis=sum_axis)
            w = W[:, :, sel].sum(axis=2).sum(axis=sum_axis)
            h1 = make_1d_hist(mass_ax, v, w)
            last_h1 = h1
            style = styler.getStyle(pairs[0][1])
            kw = dict(style.get())
            if dashed:
                kw["linestyle"] = "--"
                kw["alpha"] = 0.6
            h1.plot1d(ax=ax, label=f"{label}, {tag}", density=normalize, yerr=style.yerr, flow="none", **kw)
    labelAxis(ax, "y", last_h1.axes, label=pc.y_label)
    labelAxis(ax, "x", last_h1.axes, label=pc.x_label)
    addCMSBits(ax, [x.metadata for x in group], extra_text=f"{common_meta.get('pipeline', '')}", plot_configuration=pc)
    addLegend(ax, pc)
    ax.set_yscale(scale)
    scaleYAxis(ax)
    saveFig(fig, str(output_path), extension=pc.image_type)
    plt.close(fig)


@define
class DetectionCutMassProjection(BasePostprocessor):
    """
    Signal and QCD mass distributions (m_Hbb or m_W) from one 3D histogram, each drawn without a cut (dashed)
    and with detection >= det_cut (solid). Shows how much QCD survives and whether the cut sculpts the QCD
    mass shape toward the signal window.

    normalize=True compares shapes (each curve density-normalised); False gives weighted yields (use scale='log').
    Same grouping rule as MassWindowProjection: group WITHOUT dataset_name.
    """
    output_name: str
    signal_regex: str
    background_regex: str
    mass_axis: str = "h"          # "h" -> m_Hbb, "w" -> m_W
    det_cut: float = 0.9
    h_axis: int = 0
    w_axis: int = 1
    det_axis: int = 2
    style_set: str | StyleSet = field(factory=StyleSet)
    normalize: bool = True
    scale: str = "linear"

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeDetectionCutMassProjection, group, common_meta, output_path,
            style_set=self.style_set, signal_regex=self.signal_regex, background_regex=self.background_regex,
            h_axis=self.h_axis, w_axis=self.w_axis, det_axis=self.det_axis, mass_axis=self.mass_axis,
            det_cut=self.det_cut, normalize=self.normalize, scale=self.scale, plot_configuration=pc,
        )
