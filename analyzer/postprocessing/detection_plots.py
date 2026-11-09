from __future__ import annotations

import csv
import functools as ft
import re
from pathlib import Path
from typing import Dict, List

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


def window_mask(h_edges, w_edges, higgs_mass, w_mass, h_window, w_window, window_shape="box"):
    """(nH, nW) bool: cell centres inside the signal window.
    window_shape "box":     |m_H - higgs_mass| <= h_window  and  |m_W - w_mass| <= w_window
    window_shape "ellipse": ((m_H - higgs_mass)/h_window)^2 + ((m_W - w_mass)/w_window)^2 <= 1
                            (h_window, w_window are the semi-axes; equal values give a true circle in GeV)."""
    hc = 0.5 * (h_edges[:-1] + h_edges[1:])
    wc = 0.5 * (w_edges[:-1] + w_edges[1:])
    dh = hc[:, None] - higgs_mass
    dw = wc[None, :] - w_mass
    if window_shape == "ellipse":
        return (dh / h_window) ** 2 + (dw / w_window) ** 2 <= 1.0
    if window_shape != "box":
        raise ValueError(f"window_shape must be 'box' or 'ellipse', got {window_shape!r}")
    return (np.abs(dh) <= h_window) & (np.abs(dw) <= w_window)


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
                          plot_configuration=None, score_label="detection", window_shape="box",
                          arrays=None, x_label="m(H1 pick) [GeV]", y_label="m(W1 pick) [GeV]",
                          background_label="QCD", title_tag="", make_summary_plot=True, n_parts=2,
                          third=None, rows_per_figure=0, bkg_parts=None):
    from .plots.annotations import addCMSBits
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    if arrays is not None:
        # already-sliced (x, y, score) arrays from a scan: Vs, Ws, Vb, Wb, [x_edges, y_edges, score_edges]
        Vs, Ws, Vb, Wb, (h_edges, w_edges, d_edges) = arrays
        sig = bkg = ["scan"]
    else:
        sig, bkg, other = split_group(group, signal_regex, background_regex)
        if other:
            print(f"[DetectionBinned2D] ignoring datasets matching neither regex: {sorted(set(other))}")
        if not sig or not bkg:
            if sig or bkg:
                # e.g. a ttbar-only variant in a pipeline that has no ttbar: nothing to plot, not an error
                print(f"[DetectionBinned2D] skipping {output_path}: need signal AND background in one group "
                      f"(found {len(sig)} signal, {len(bkg)} background) [{common_meta.get('pipeline', '')}]")
                return None
            raise RuntimeError(
                f"need signal AND background in one group (found {len(sig)} signal, {len(bkg)} background). Group on "
                f"{{pipeline, name}} without dataset_name so signal and QCD land together; check signal_regex "
                f"{signal_regex!r} / background_regex {background_regex!r} against meta['dataset_name'].")
        Vs, Ws, (h_edges, w_edges, d_edges) = hist_arrays(sum_histograms(sig), h_axis, w_axis, det_axis)
        Vb, Wb, e2 = hist_arrays(sum_histograms(bkg), h_axis, w_axis, det_axis)
        if not all(np.array_equal(a, b) for a, b in zip((h_edges, w_edges, d_edges), e2)):
            raise RuntimeError("signal and background histograms have different axes")
    n_det = len(d_edges) - 1
    win = window_mask(h_edges, w_edges, higgs_mass, w_mass, h_window, w_window, window_shape)

    # detection ranges (grid rows), snapped to axis edges
    wanted = sorted(set(det_bin_edges))
    snapped = []
    for x in wanted:
        i, exact = snap_edge(d_edges, x)
        if not exact:
            print(f"[DetectionBinned2D] {score_label} edge {x:g} is not an edge of the score axis; using {d_edges[i]:g}")
        snapped.append(i)
    snapped = sorted(set(snapped))
    ranges = list(zip(snapped[:-1], snapped[1:]))
    results = [((d_edges[a], d_edges[b]), region_stats(Vs, Ws, Vb, Wb, slice(a, b), signal_scale, bkg_scale, win,
                                                        min_b_events)) for a, b in ranges]
    # cumulative scan at EVERY detection bin edge: keep detection >= cut
    cum = [(d_edges[i], region_stats(Vs, Ws, Vb, Wb, slice(i, n_det), signal_scale, bkg_scale, win, min_b_events))
           for i in range(n_det)]

    fmt = lambda v, p=2: "  -  " if (v is None or not np.isfinite(v)) else f"{v:.{p}f}"
    title_extra = f"{common_meta.get('pipeline', '')}" + (f"  {title_tag}" if title_tag else "")
    print(f"\n=== {score_label}-binned mass plane{(' ' + title_tag) if title_tag else ''}: signal {len(sig)} dataset(s), background {len(bkg)}; "
          f"scales signal x{signal_scale:g}, background x{bkg_scale:g}; {window_shape} window H {higgs_mass:g}+/-{h_window:g}, "
          f"W {w_mass:g}+/-{w_window:g}; min n_eff(B) {min_b_events:g} ===")
    hdr = f"  {score_label[:13]:>13s} {'S':>10s} {'B':>10s} {'nEffB':>8s} {'S/sqrtB all':>13s} {'in window':>20s} {'quad.':>9s}"
    print(hdr)
    for (lo, hi), s in results:
        print(f"  {f'[{lo:.2f},{hi:.2f}' + (']' if hi == d_edges[-1] else ')'):>13s} {s['S']:10.3g} {s['B']:10.3g} "
              f"{s['neff_B']:8.1f} {fmt(s['z_all'], 4):>13s} "
              f"{(fmt(s['z_window'], 4) + ' +/- ' + fmt(s['z_window_err'], 4)):>20s} {fmt(s['z_quad'], 4):>9s}")
    # total over every (score bin x mass cell): the bins are disjoint, so combine in quadrature
    zq = [s['z_quad'] for _, s in results if np.isfinite(s['z_quad'])]
    z_total = float("nan")
    if zq:
        z_total = float(np.sqrt(np.sum(np.square(zq))))
        n_cells = int(sum(s['n_cells_used'] for _, s in results))
        print(f"  TOTAL over all {len(zq)} {score_label} bins and {n_cells} unmasked cells: "
              f"sqrt(sum S^2/B) = {z_total:.4f}   (S/sqrt(B) = 1 would be ~1 sigma)")
    best = max(((c, s) for c, s in cum if np.isfinite(s["z_window"])), key=lambda cs: cs[1]["z_window"], default=None)
    if best:
        print(f"  best cumulative cut by in-window S/sqrt(B): {score_label} >= {best[0]:.3f} "
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

    # ---- grid: rows = detection ranges; signal | background | S/sqrt(B) per cell; split in two halves ----
    n = len(results)
    if rows_per_figure and rows_per_figure > 0:
        n_parts = -(-n // int(rows_per_figure))
    n_parts = max(1, min(n_parts, n))
    per = -(-n // n_parts)  # ceil
    parts = list(bkg_parts.items()) if bkg_parts and len(bkg_parts) > 1 else []   # separate background panels
    n_cols = 2 + max(1, len(parts)) + (1 if third is not None else 0)
    z_col = 1 + max(1, len(parts))
    for part in range(n_parts):
        idx = list(range(part * per, min(n, (part + 1) * per)))
        if not idx:
            continue
        fig, axes = plt.subplots(len(idx), n_cols, figsize=(8.7 * n_cols, 7.2 * len(idx)), squeeze=False,
                                 layout="constrained")
        for r, i in enumerate(idx):
            (lo, hi), s = results[i]
            rng_txt = f"{score_label} [{lo:.2f}, {hi:.2f}{']' if i == n - 1 else ')'}"
            a_, b_ = ranges[i]
            panels = [("signal", s["S2"], "viridis", f"S = {s['S']:.3g}", False)]
            if parts:
                for pname, (pv, pw) in parts:
                    arr_p = pv[:, :, a_:b_].sum(axis=2) * bkg_scale
                    var_p = pw[:, :, a_:b_].sum(axis=2) * bkg_scale ** 2
                    panels.append((pname, arr_p, "magma",
                                   f"B = {arr_p.sum():.3g}, n_eff = {_neff(arr_p.sum(), var_p.sum()):.0f}", True))
            else:
                panels.append((background_label, s["B2"], "magma", f"B = {s['B']:.3g}, n_eff = {s['neff_B']:.0f}", True))
            panels.append(("S/sqrt(B) per cell" + (f" (total: {background_label})" if parts else ""),
                           s["zmap"], "plasma", "", False))
            for j, (ttl, arr, cmap, extra, mask_zero) in enumerate(panels):
                ax = axes[r][j]
                data = np.ma.masked_invalid(arr).T
                if mask_zero:
                    data = np.ma.masked_where(data <= 0, data)
                if np.ma.count(data) == 0:
                    ax.set_xlim(h_edges[0], h_edges[-1]); ax.set_ylim(w_edges[0], w_edges[-1])
                    ax.text(0.5, 0.5, "no events" if mask_zero else "no cell passes\nn_eff(B) cut", transform=ax.transAxes,
                            ha="center", va="center", fontsize=20, color="0.5")
                else:
                    mesh = ax.pcolormesh(h_edges, w_edges, data, cmap=cmap, shading="flat")
                    fig.colorbar(mesh, ax=ax)
                ax.set_title(f"{rng_txt} -- {ttl}" + (f"\n{extra}" if extra else ""), fontsize=17)
                if j == 0:
                    if window_shape == "ellipse":
                        from matplotlib.patches import Ellipse
                        ax.add_patch(Ellipse((higgs_mass, w_mass), 2 * h_window, 2 * w_window,
                                             fill=False, edgecolor="w", linestyle="--", linewidth=1.5))
                    else:
                        ax.add_patch(plt.Rectangle((higgs_mass - h_window, w_mass - w_window), 2 * h_window, 2 * w_window,
                                                   fill=False, edgecolor="w", linestyle="--", linewidth=1.5))
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            if third is not None:
                # 4th column: projection of the third mass axis after the plane-window cuts, for THIS score range
                a, b = ranges[i]
                ax = axes[r][z_col + 1]
                te = third["edges"]
                Sx = third["vs"][:, a:b].sum(axis=1) * signal_scale
                vSx = third["ws"][:, a:b].sum(axis=1) * signal_scale ** 2
                Bx = third["vb"][:, a:b].sum(axis=1) * bkg_scale
                vBx = third["wb"][:, a:b].sum(axis=1) * bkg_scale ** 2
                dsc = third["display_scale"]
                ax.stairs(Sx * dsc, te, color=SAMPLE_COLORS["signal"], linewidth=2.0,
                          label=f"signal x{dsc:g}" if dsc != 1 else "signal")
                tparts = third.get("parts") or {}
                curves_pos = [(Sx * dsc)[Sx * dsc > 0]]
                if len(tparts) > 1:
                    # each background separately, plus the total (dashed grey) that the S/sqrt(B) uses
                    for pname, (pvx, _) in tparts.items():
                        yb = pvx[:, a:b].sum(axis=1) * bkg_scale
                        ax.stairs(yb, te, color=SAMPLE_COLORS.get(pname, "0.4"), linewidth=2.0, label=pname)
                        curves_pos.append(yb[yb > 0])
                    ax.stairs(Bx, te, color="0.3", linewidth=1.5, linestyle="--", label="total bkg")
                else:
                    ax.stairs(Bx, te, color=SAMPLE_COLORS["QCD"], linewidth=2.0, label=background_label)
                curves_pos.append(Bx[Bx > 0])
                c_, hw_ = third["center"], third["hw"]
                ax.axvspan(c_ - hw_, c_ + hw_, color="0.85", alpha=0.5, zorder=0)
                pos = np.concatenate(curves_pos)
                ax.set_yscale(third["scale"])
                if pos.size:
                    if third["scale"] == "log":
                        ax.set_ylim(pos.min() * 0.5, pos.max() * 30)
                    else:
                        ax.set_ylim(0, pos.max() * 1.6)
                tc = 0.5 * (te[:-1] + te[1:])
                inw = np.abs(tc - c_) <= hw_
                zw, zwe = _z(float(Sx[inw].sum()), float(vSx[inw].sum()), float(Bx[inw].sum()), float(vBx[inw].sum()),
                             _neff(Bx[inw].sum(), vBx[inw].sum()), min_b_events)
                ztxt = "-" if not np.isfinite(zw) else f"{zw:.3f}"
                ax.text(0.97, 0.95, f"in window: S/sqrt(B) = {ztxt}\nS = {Sx.sum():.3g}, B = {Bx.sum():.3g}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=15,
                        bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.85))
                ax.set_title(f"{rng_txt} -- m({third['label']})\n[{third['cuts_txt']}]", fontsize=17)
                ax.set_xlabel(f"m({third['label']}) [GeV]")
                ax.set_ylabel("weighted events / bin")
                if r == 0:
                    ax.legend(loc="upper left", fontsize=14, frameon=False)
        fig.suptitle(f"{title_extra}  Mass plane by {score_label}-score range (part {part + 1}/{n_parts}). "
                     f"Dashed {'ellipse' if window_shape == 'ellipse' else 'box'} = window; S/sqrt(B) cells need n_eff(B) >= {min_b_events:g}", fontsize=19)
        out = str(output_path) if n_parts == 1 else f"{stem}_part{part + 1}{ext}"
        saveFig(fig, out, extension=pc.image_type)
        plt.close(fig)

    summary = {"z_total": z_total, "n_cells": int(sum(s['n_cells_used'] for _, s in results)),
               "best_cut": float(best[0]) if best else float("nan"),
               "best_z_window": float(best[1]["z_window"]) if best else float("nan"),
               "S_total": float(cum[0][1]["S"]), "B_total": float(cum[0][1]["B"])}
    if not make_summary_plot:
        return summary

    # ---- summary: significance and efficiencies vs cumulative detection cut ----
    fig, (ax_z, ax_e) = plt.subplots(1, 2, figsize=(22, 8), layout="constrained")
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
    ax_z.set_xlabel(f"{score_label}-score cut (keep >= cut)")
    ax_z.set_ylabel("S/sqrt(B)")
    ax_z.set_ylim(0, ax_z.get_ylim()[1] * 1.9)
    ax_z.legend(loc="upper right", fontsize=16)
    s0, b0 = cum[0][1]["S"], cum[0][1]["B"]
    ax_e.plot(cx, [s["S"] / s0 if s0 else np.nan for _, s in cum], color="#2a9959", label="signal kept")
    ax_e.plot(cx, [s["B"] / b0 if b0 else np.nan for _, s in cum], color="#c0392b", label=f"{background_label} kept")
    ax_e.set_yscale("log")
    ax_e.set_xlabel(f"{score_label}-score cut (keep >= cut)")
    ax_e.set_ylabel("fraction kept (weighted)")
    ax_e.legend(fontsize=16, loc="lower left")
    addCMSBits(ax_z, [x.metadata for x in group], extra_text=title_extra, plot_configuration=pc)
    saveFig(fig, f"{stem}_summary{ext}", extension=pc.image_type)
    plt.close(fig)
    return summary


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
    higgs_mass, w_mass, h_window, w_window : the window for the in-window significance
                                     (half-widths of a box, or semi-axes of an ellipse).
    window_shape                   : "box" (default) or "ellipse".
    score_label                    : name of the score on the third axis (default "detection").
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
    window_shape: str = "box"           # "box" or "ellipse" (h_window / w_window = semi-axes)
    score_label: str = "detection"      # what the third axis is, e.g. "classifier"
    background_label: str = "QCD"       # name shown for the background panels (e.g. "background" once ttbar is included)

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
            score_label=self.score_label, window_shape=self.window_shape,
            background_label=self.background_label,
        )


# ----------------------------------------------------------------------------------------------
# plot 2: mass projection before / after a detection cut, signal vs QCD
# ----------------------------------------------------------------------------------------------
def makeDetectionCutMassProjection(group, common_meta, output_path, style_set, signal_regex, background_regex,
                                   h_axis, w_axis, det_axis, mass_axis, det_cut, normalize, scale,
                                   plot_configuration=None, score_label="detection", samples=None,
                                   signal_label="signal", display_signal_scale=1.0):
    from .plots.annotations import addCMSBits, labelAxis
    from .plots.utils import addLegend, saveFig, scaleYAxis
    from .plots.common import PlotConfiguration
    from .style import Styler

    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    if samples:
        # one curve pair per sample label: the signal and EVERY background present (QCD and ttbar drawn separately)
        by_label, unmatched = assign_samples(group, samples)
        if unmatched:
            print(f"[DetectionCutMassProjection] ignoring datasets matching no sample pattern: {sorted(set(unmatched))}")
        series = [(l, by_label[l]) for l in [signal_label] + [k for k in samples if k != signal_label] if by_label.get(l)]
        if not by_label.get(signal_label) or len(series) < 2:
            print(f"[DetectionCutMassProjection] skipping {output_path}: need signal AND at least one background "
                  f"(found { {l: len(v) for l, v in by_label.items()} }) [{common_meta.get('pipeline', '')}]")
            return
    else:
        sig, bkg, other = split_group(group, signal_regex, background_regex)
        if other:
            print(f"[DetectionCutMassProjection] ignoring datasets matching neither regex: {sorted(set(other))}")
        if not sig or not bkg:
            raise RuntimeError(f"need signal AND background in one group (found {len(sig)} signal, {len(bkg)} background).")
        series = [("signal", sig), ("QCD", bkg)]
    if mass_axis not in ("h", "w"):
        raise RuntimeError("mass_axis must be 'h' (m_Hbb) or 'w' (m_W)")
    fig, ax = plt.subplots()
    last_h1 = None
    cut_idx = None
    for label, pairs in series:
        V, W, (h_edges, w_edges, d_edges) = hist_arrays(sum_histograms(pairs), h_axis, w_axis, det_axis)
        if cut_idx is None:
            cut_idx, exact = snap_edge(d_edges, det_cut)
            if not exact:
                print(f"[DetectionCutMassProjection] cut {det_cut:g} is not a {score_label}-axis edge; using {d_edges[cut_idx]:g}")
        src = pairs[0][0].histogram
        mass_ax = src.axes[h_axis if mass_axis == "h" else w_axis]
        sum_axis = 1 if mass_axis == "h" else 0     # sum the OTHER mass axis away
        for sel, tag, dashed in ((slice(0, None), "no cut", True), (slice(cut_idx, None), f"{score_label} >= {d_edges[cut_idx]:g}", False)):
            v = V[:, :, sel].sum(axis=2).sum(axis=sum_axis)
            w = W[:, :, sel].sum(axis=2).sum(axis=sum_axis)
            shown = label
            if samples and label == signal_label and not normalize and display_signal_scale != 1:
                v, w = v * display_signal_scale, w * display_signal_scale ** 2
                shown = f"{label} x{display_signal_scale:g}"
            h1 = make_1d_hist(mass_ax, v, w)
            last_h1 = h1
            if samples:
                kw = dict(color=SAMPLE_COLORS.get(label, "0.4"), linewidth=1.8)
                yerr = True
            else:
                style = styler.getStyle(pairs[0][1])
                kw = dict(style.get())
                yerr = style.yerr
            if dashed:
                kw["linestyle"] = "--"
                kw["alpha"] = 0.6
            h1.plot1d(ax=ax, label=f"{shown}, {tag}", density=normalize, yerr=yerr, flow="none", **kw)
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

    samples (optional): {label: regex on dataset_name}, e.g. {signal: "ggHH", QCD: "(?i)qcd", ttbar: "(?i)tt_(hadronic)"}.
    With it, the signal and every background that is present are drawn as separate curves (signal_regex /
    background_regex are then ignored); a group missing the signal or every background is skipped with a message.
    """
    output_name: str
    signal_regex: str = ""
    background_regex: str = ""
    samples: Dict[str, str] | None = None    # {label: dataset_name regex}: signal + EACH background drawn separately
    signal_label: str = "signal"
    display_signal_scale: float = 1.0        # only with samples and normalize False
    mass_axis: str = "h"          # "h" -> m_Hbb, "w" -> m_W
    det_cut: float = 0.9
    h_axis: int = 0
    w_axis: int = 1
    det_axis: int = 2
    style_set: str | StyleSet = field(factory=StyleSet)
    normalize: bool = True
    scale: str = "linear"
    score_label: str = "detection"

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeDetectionCutMassProjection, group, common_meta, output_path,
            style_set=self.style_set, signal_regex=self.signal_regex, background_regex=self.background_regex,
            h_axis=self.h_axis, w_axis=self.w_axis, det_axis=self.det_axis, mass_axis=self.mass_axis,
            det_cut=self.det_cut, normalize=self.normalize, scale=self.scale, plot_configuration=pc,
            score_label=self.score_label, samples=self.samples, signal_label=self.signal_label,
            display_signal_scale=self.display_signal_scale,
        )


# ----------------------------------------------------------------------------------------------
# plot 3: compare several scores (e.g. detection vs classifier) at fixed signal efficiency
# ----------------------------------------------------------------------------------------------
def cumulative_curve(vs, ws, vb, wb, signal_scale, bkg_scale, min_b):
    """
    Cut-and-count curve of ONE score from 1D weight sums per score bin (signal vs, ws variances;
    background vb, wb). Point i keeps score bins i.. (cut = lower edge of bin i). Returns arrays
    over the cut position:
      eff_S, eff_B  : kept fraction of the weighted signal / background
      rejection     : 1 / eff_B                     (nan where B is empty or n_eff(B) < min_b)
      z, z_err      : S/sqrt(B) over ALL events kept (nan where n_eff(B) < min_b)
      S, B, neff_B  : the weighted yields and the effective background event count
    """
    cs = lambda a: np.cumsum(np.asarray(a, dtype=float)[::-1])[::-1]
    S, varS = cs(vs) * signal_scale, cs(ws) * signal_scale ** 2
    b_u, bv_u = cs(vb), cs(wb)
    B, varB = b_u * bkg_scale, bv_u * bkg_scale ** 2
    S0, B0 = S[0], B[0]
    n = len(S)
    neff = np.array([_neff(b_u[i], bv_u[i]) for i in range(n)])
    eff_S = S / S0 if S0 > 0 else np.full(n, np.nan)
    eff_B = B / B0 if B0 > 0 else np.full(n, np.nan)
    usable = (B > 0) & (neff >= min_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        rejection = np.where(usable, 1.0 / eff_B, np.nan)
    zz = [_z(S[i], varS[i], B[i], varB[i], neff[i], min_b) for i in range(n)]
    return {"eff_S": eff_S, "eff_B": eff_B, "rejection": rejection,
            "z": np.array([a for a, _ in zz]), "z_err": np.array([b for _, b in zz]),
            "S": S, "B": B, "neff_B": neff}


def _auc(eff_b, eff_s):
    """Area under the ROC (signal efficiency vs background efficiency), trapezoid rule over the cut scan,
    closed with the (0, 0) and (1, 1) corners. Weighted: eff_b / eff_s are fractions of the weighted yields."""
    fpr = np.concatenate([[0.0], np.asarray(eff_b, dtype=float)[::-1], [1.0]])
    tpr = np.concatenate([[0.0], np.asarray(eff_s, dtype=float)[::-1], [1.0]])
    ok = np.isfinite(fpr) & np.isfinite(tpr)
    fpr, tpr = fpr[ok], tpr[ok]
    order = np.argsort(fpr, kind="stable")
    return float(np.trapezoid(tpr[order], fpr[order]) if hasattr(np, "trapezoid") else np.trapz(tpr[order], fpr[order]))


def at_signal_efficiency(curve, target):
    """Index of the TIGHTEST cut that still keeps at least `target` of the signal (None if none does)."""
    ok = np.where(np.isfinite(curve["eff_S"]) & (curve["eff_S"] >= target))[0]
    return int(ok[-1]) if len(ok) else None


def makeScoreComparison(group, common_meta, output_path, signal_regex, background_regex, scores, name_key,
                        signal_scale, bkg_scale, min_b_events, target_signal_effs, plot_configuration=None):
    from .plots.annotations import addCMSBits
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    available = sorted({str(x[1].get(name_key, "<missing>")) for x in group})
    if all(name_key not in x[1] for x in group):
        raise RuntimeError(f"meta has no key {name_key!r} to tell the score histograms apart; "
                           f"available meta keys: {sorted(group[0][1].keys())}")
    curves = {}
    for label, name_regex in scores.items():
        chosen = [x for x in group if re.search(name_regex, str(x[1].get(name_key, ""))) is not None]
        if not chosen:
            raise RuntimeError(f"no histogram in the group matches {name_regex!r} (for score {label!r}); "
                               f"histogram names present: {available}")
        sig, bkg, _ = split_group(chosen, signal_regex, background_regex)
        if not sig or not bkg:
            print(f"[ScoreComparison] skipping score {label!r}: need signal AND background (found {len(sig)} signal, "
                  f"{len(bkg)} background) among histograms matching {name_regex!r} "
                  f"[{common_meta.get('pipeline', '')}]")
            continue
        hs, hb = sum_histograms(sig), sum_histograms(bkg)
        for h in (hs, hb):
            if h.ndim != 1:
                raise RuntimeError(f"score {label!r}: expected 1D histograms, got {h.ndim}D "
                                   f"({[a.name for a in h.axes]})")
        if not np.array_equal(hs.axes[0].edges, hb.axes[0].edges):
            raise RuntimeError(f"score {label!r}: signal and background histograms have different axes")
        vs, ws, _ = _hist1d_arrays(hs)
        vb, wb, _ = _hist1d_arrays(hb)
        edges = np.asarray(hs.axes[0].edges, dtype=float)
        c = cumulative_curve(vs, ws, vb, wb, signal_scale, bkg_scale, min_b_events)
        c["cut"] = edges[:-1]
        c["auc"] = _auc(c["eff_B"], c["eff_S"])
        curves[label] = c
    if not curves:
        print(f"[ScoreComparison] nothing to plot for {output_path}: no score had both signal and background")
        return

    # ---- printed comparison at fixed signal efficiency ----
    print(f"\n=== Score comparison at fixed signal efficiency (rejection = 1/eff_B; S/sqrt(B) over all events; "
          f"'-' = fewer than {min_b_events:g} effective QCD events left at that cut) ===")
    labels = list(curves)
    w = max(12, max(len(l) for l in labels))
    print(f"  {'sig eff':>8s}" + "".join(f" | {l:>{w}s}: {'cut':>6s} {'rejection':>10s} {'S/sqrtB':>8s}" for l in labels))
    for t in target_signal_effs:
        cells = []
        for l in labels:
            c = curves[l]
            i = at_signal_efficiency(c, t)
            if i is None:
                cells.append(f" | {'':>{w}s}  {'-':>6s} {'-':>10s} {'-':>8s}")
                continue
            rej, z = c["rejection"][i], c["z"][i]
            cells.append(f" | {'':>{w}s}  {c['cut'][i]:6.3f} "
                         f"{('-' if not np.isfinite(rej) else f'{rej:.4g}'):>10s} "
                         f"{('-' if not np.isfinite(z) else f'{z:.4f}'):>8s}")
        print(f"  {t:8.2f}" + "".join(cells))

    print("  weighted ROC AUC (signal eff vs QCD eff, QCD weighted by cross section): "
          + ", ".join(f"{l}: {curves[l]['auc']:.4f}" for l in labels))

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    cols = ["score", "cut", "eff_S", "eff_B", "rejection", "S", "B", "neff_B", "z", "z_err"]
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for l in labels:
            c = curves[l]
            for i in range(len(c["cut"])):
                wr.writerow({"score": l, **{k: c[k][i] for k in cols[1:] if k != "cut"}, "cut": c["cut"][i]})

    # ---- plot: rejection vs signal efficiency | S/sqrt(B) vs signal efficiency ----
    fig, (ax_r, ax_z) = plt.subplots(1, 2, figsize=(22, 8), layout="constrained")
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    for k, l in enumerate(labels):
        c = curves[l]
        col = palette[k % len(palette)]
        m = np.isfinite(c["eff_S"]) & np.isfinite(c["rejection"]) & (c["eff_S"] > 0)
        ax_r.plot(c["eff_S"][m], c["rejection"][m], color=col, label=l)
        mz = np.isfinite(c["eff_S"]) & np.isfinite(c["z"]) & (c["eff_S"] > 0)
        ax_z.plot(c["eff_S"][mz], c["z"][mz], color=col, label=l)
        ok = mz & np.isfinite(c["z_err"])
        ax_z.fill_between(c["eff_S"][ok], (c["z"] - c["z_err"])[ok], (c["z"] + c["z_err"])[ok],
                          color=col, alpha=0.2, linewidth=0)
    ax_r.set_yscale("log")
    ax_r.set_ylim(top=ax_r.get_ylim()[1] * 30)   # headroom: the curves start top-left, where the CMS text goes
    ax_r.set_xlim(0, 1)
    ax_z.set_xlim(0, 1)
    ax_r.set_xlabel("signal efficiency")
    ax_r.set_ylabel("QCD rejection (1 / background efficiency)")
    ax_z.set_xlabel("signal efficiency")
    ax_z.set_ylabel("S/sqrt(B), all events")
    ax_z.set_ylim(0, ax_z.get_ylim()[1] * 1.9)
    ax_r.legend(loc="upper right", fontsize=16)
    ax_z.legend(loc="upper right", fontsize=16)
    addCMSBits(ax_r, [x.metadata for x in group], extra_text=f"{common_meta.get('pipeline', '')}", plot_configuration=pc)
    saveFig(fig, str(output_path), extension=pc.image_type)
    plt.close(fig)

    # ---- ROC (signal efficiency vs QCD efficiency), linear and log-x, weighted AUC in the legend ----
    fig, (ax_l, ax_g) = plt.subplots(1, 2, figsize=(22, 9), layout="constrained")
    for k, l in enumerate(labels):
        c = curves[l]
        col = palette[k % len(palette)]
        m = np.isfinite(c["eff_S"]) & np.isfinite(c["eff_B"]) & (c["neff_B"] >= min_b_events)
        lab = f"{l} (AUC = {c['auc']:.3f})"
        ax_l.plot(c["eff_B"][m], c["eff_S"][m], color=col, label=lab)
        mg = m & (c["eff_B"] > 0)
        ax_g.plot(c["eff_B"][mg], c["eff_S"][mg], color=col, label=lab)
    ax_l.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1.25)
    ax_g.set_xscale("log")
    ax_g.set_ylim(0, 1.25)
    for ax in (ax_l, ax_g):
        ax.set_xlabel("QCD efficiency (false positive rate, cross-section weighted)")
        ax.set_ylabel("signal efficiency (true positive rate)")
        ax.legend(loc="upper right", fontsize=16)
    addCMSBits(ax_l, [x.metadata for x in group], extra_text=f"{common_meta.get('pipeline', '')}", plot_configuration=pc)
    saveFig(fig, f"{stem}_roc{ext}", extension=pc.image_type)
    plt.close(fig)


def _hist1d_arrays(h):
    """(values, variances, edges) of a 1D histogram, no flow bins; Double storage counts as Poisson."""
    v = np.asarray(h.values(), dtype=float)
    var = h.variances()
    var = v.copy() if var is None else np.asarray(var, dtype=float)
    return v, var, np.asarray(h.axes[0].edges, dtype=float)


@define
class ScoreComparison(BasePostprocessor):
    """
    Compare several per-event scores (e.g. the H1 detection score and the classifier's signal probability)
    on equal footing: for each score, QCD rejection (1 / background efficiency) against signal efficiency,
    and S/sqrt(B) over all events against signal efficiency, from a cumulative cut (keep score >= cut).
    A better score sits higher in both panels. Also prints a table at target_signal_effs and writes
    <output_name minus extension>.csv with the full curves.

    Input: one FINE-binned 1D histogram per score and sample (1000 bins on [0, 1] is plenty), so the
    cut can sit close to 1. Group WITHOUT dataset_name and without the histogram name (e.g.
    group {pipeline, era.name}), so signal and QCD and every score's histogram land in one group;
    `scores` then picks the histograms apart by regex on meta[name_key] (the histogram name), and
    signal_regex / background_regex pick the samples apart by meta["dataset_name"].

    scores              : {legend label: regex on the histogram name}, e.g.
                          {"H1 detection": "^roc_H1_detection$", "classifier": "^roc_classification_signal$"}
    min_b_events        : points whose cumulative effective QCD event count is below this are not drawn.
    target_signal_effs  : signal efficiencies of the printed comparison table.
    """
    output_name: str
    signal_regex: str
    background_regex: str
    scores: Dict[str, str]
    name_key: str = "name"
    signal_scale: float = 1.0
    bkg_scale: float = 1.0
    min_b_events: float = 5.0
    target_signal_effs: List[float] = field(factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.8])

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeScoreComparison, group, common_meta, output_path,
            signal_regex=self.signal_regex, background_regex=self.background_regex, scores=self.scores,
            name_key=self.name_key, signal_scale=self.signal_scale, bkg_scale=self.bkg_scale,
            min_b_events=self.min_b_events, target_signal_effs=self.target_signal_effs, plot_configuration=pc,
        )


def makeROCCurve(group, common_meta, output_path, signal_regex, background_regex, scores, name_key,
                 signal_scale, bkg_scale, min_b_events, plot_configuration=None):
    """One figure with the weighted ROC (signal efficiency vs QCD efficiency) of every score in `scores`,
    linear (left) and log-x (right) panels, weighted AUC in each legend entry. Also writes <stem>.csv."""
    from .plots.annotations import addCMSBits
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    available = sorted({str(x[1].get(name_key, "<missing>")) for x in group})
    curves = {}
    for label, name_regex in scores.items():
        chosen = [x for x in group if re.search(name_regex, str(x[1].get(name_key, ""))) is not None]
        if not chosen:
            raise RuntimeError(f"ROCCurve: no histogram matches {name_regex!r} (score {label!r}); "
                               f"histogram names present: {available}")
        sig, bkg, _ = split_group(chosen, signal_regex, background_regex)
        if not sig or not bkg:
            print(f"[ROCCurve] skipping score {label!r}: needs signal AND background "
                  f"(found {len(sig)} signal, {len(bkg)} background) [{common_meta.get('pipeline', '')}]")
            continue
        hs, hb = sum_histograms(sig), sum_histograms(bkg)
        if hs.ndim != 1 or hb.ndim != 1:
            raise RuntimeError(f"ROCCurve: score {label!r}: expected 1D histograms")
        vs, ws, _ = _hist1d_arrays(hs)
        vb, wb, _ = _hist1d_arrays(hb)
        edges = np.asarray(hs.axes[0].edges, dtype=float)
        c = cumulative_curve(vs, ws, vb, wb, signal_scale, bkg_scale, min_b_events)
        c["cut"] = edges[:-1]
        c["auc"] = _auc(c["eff_B"], c["eff_S"])
        curves[label] = c
    if not curves:
        print(f"[ROCCurve] nothing to plot for {output_path}: no score had both signal and background")
        return
    labels = list(curves)
    print("\n=== ROC (weighted; QCD weighted by cross section) ===  AUC: "
          + ", ".join(f"{l}: {curves[l]['auc']:.4f}" for l in labels))

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["score", "cut", "eff_S", "eff_B", "neff_B"])
        for l in labels:
            c = curves[l]
            for i in range(len(c["cut"])):
                wr.writerow([l, c["cut"][i], c["eff_S"][i], c["eff_B"][i], c["neff_B"][i]])

    fig, (ax_l, ax_g) = plt.subplots(1, 2, figsize=(22, 9), layout="constrained")
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    for k, l in enumerate(labels):
        c = curves[l]
        col = palette[k % len(palette)]
        m = np.isfinite(c["eff_S"]) & np.isfinite(c["eff_B"]) & (c["neff_B"] >= min_b_events)
        lab = f"{l} (AUC = {c['auc']:.3f})"
        ax_l.plot(c["eff_B"][m], c["eff_S"][m], color=col, label=lab)
        mg = m & (c["eff_B"] > 0)
        ax_g.plot(c["eff_B"][mg], c["eff_S"][mg], color=col, label=lab)
    ax_l.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1.25)
    ax_g.set_xscale("log")
    ax_g.set_ylim(0, 1.25)
    for ax in (ax_l, ax_g):
        ax.set_xlabel("QCD efficiency (false positive rate, cross-section weighted)")
        ax.set_ylabel("signal efficiency (true positive rate)")
        ax.legend(loc="upper right", fontsize=16)
    addCMSBits(ax_l, [x.metadata for x in group], extra_text=f"{common_meta.get('pipeline', '')}", plot_configuration=pc)
    saveFig(fig, str(output_path), extension=pc.image_type)
    plt.close(fig)


@define
class ROCCurve(BasePostprocessor):
    """
    ROC curves (signal efficiency vs QCD efficiency, cumulative cut "score >= cut", weighted AUC in the
    legend) for one or more per-event scores, in their own output file: <output_name> (png) plus
    <output_name minus extension>.csv with the curve points.

    Same input layout as ScoreComparison: FINE-binned 1D score histograms (roc_*), signal and QCD in one
    group (group {pipeline, era.name}, NOT dataset_name or name); `scores` = {legend label: regex on the
    histogram name}. Several ROCCurve entries with different `scores` give separate files.
    """
    output_name: str
    signal_regex: str
    background_regex: str
    scores: Dict[str, str]
    name_key: str = "name"
    signal_scale: float = 1.0
    bkg_scale: float = 1.0
    min_b_events: float = 5.0

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeROCCurve, group, common_meta, output_path,
            signal_regex=self.signal_regex, background_regex=self.background_regex, scores=self.scores,
            name_key=self.name_key, signal_scale=self.signal_scale, bkg_scale=self.bkg_scale,
            min_b_events=self.min_b_events, plot_configuration=pc,
        )


# ----------------------------------------------------------------------------------------------
# plot 4: cut + project an N-D histogram (e.g. the 4D (m_Hbb, m_W1, m_HWW, score) fit histogram) down to 1D / 2D
# ----------------------------------------------------------------------------------------------
SAMPLE_COLORS = {"signal": "#2a9959", "QCD": "#c0392b", "ttbar": "#2980b9"}


def assign_samples(group, samples):
    """(item, meta) pairs -> {label: [pairs]} by regex on meta['dataset_name']. A dataset matching two labels is an
    error (the patterns must be disjoint); datasets matching none are returned separately and ignored by the caller."""
    out = {label: [] for label in samples}
    unmatched = []
    for x in group:
        name = str(x[1].get("dataset_name", ""))
        hits = [label for label, rx in samples.items() if re.search(rx, name) is not None]
        if len(hits) > 1:
            raise RuntimeError(f"dataset_name {name!r} matches several sample patterns {hits}; make them disjoint.")
        if hits:
            out[hits[0]].append(x)
        else:
            unmatched.append(name)
    return out, unmatched


def resolve_axis(h, key):
    """Axis index from `key`: an integer (as int or digit string) is the axis position; anything else is a regex
    searched in the axis name and label, which must pick exactly one axis."""
    s = str(key).strip()
    if s.lstrip("-").isdigit():
        i = int(s)
        if not -h.ndim <= i < h.ndim:
            raise RuntimeError(f"axis index {i} is out of range for a {h.ndim}D histogram")
        return i % h.ndim
    hits = [i for i, a in enumerate(h.axes)
            if re.search(s, str(a.name)) is not None or re.search(s, str(getattr(a, 'label', ''))) is not None]
    if len(hits) != 1:
        raise RuntimeError(f"axis key {key!r} matched {len(hits)} axes; axes are "
                           f"{[(i, a.name) for i, a in enumerate(h.axes)]} -- use a more specific regex or the axis index")
    return hits[0]


def _bound_index(edges, x, side):
    """Edge index for a cut bound; None / +-inf mean 'open'. Warns when x is not an axis edge (snaps to nearest)."""
    if x is None or not np.isfinite(x):
        return (0 if side == "lo" else len(edges) - 1), True
    return snap_edge(edges, float(x))


def slice_project(h, cuts, keep, warn=True):
    """
    Cut and project an N-D histogram.
      cuts : {axis key: [lo, hi]}   value range kept on that axis ([lo, hi) in bin terms; None / +-inf = open),
                                    snapped to bin edges (a warning is printed if a bound is not an edge)
      keep : [axis key, ...]        1 or 2 axes that stay; every other axis is summed over (after its cut, if any)
    Returns (values, variances, [edges of each kept axis], {axis name: (lo, hi) actually applied}, [kept axis indices]).
    No flow bins. Double-storage histograms (no variances) are treated as Poisson counts. The kept axes come back in
    the order of `keep`.
    """
    if not 1 <= len(keep) <= h.ndim:
        raise RuntimeError("keep must name at least one axis")
    keep_idx = [resolve_axis(h, k) for k in keep]
    if len(set(keep_idx)) != len(keep_idx):
        raise RuntimeError(f"keep names the same axis twice: {keep}")
    v = np.asarray(h.values(), dtype=float)
    var = h.variances()
    var = v.copy() if var is None else np.asarray(var, dtype=float)

    slices = [slice(None)] * h.ndim
    applied = {}
    for key, bounds in cuts.items():
        i = resolve_axis(h, key)
        lo, hi = (list(bounds) + [None, None])[:2]
        edges = np.asarray(h.axes[i].edges, dtype=float)
        a, a_exact = _bound_index(edges, lo, "lo")
        b, b_exact = _bound_index(edges, hi, "hi")
        for x, exact, idx in ((lo, a_exact, a), (hi, b_exact, b)):
            if not exact and warn:
                print(f"[SliceProjection] cut {x:g} on axis '{h.axes[i].name}' is not a bin edge; using {edges[idx]:g}")
        if b <= a:
            raise RuntimeError(f"empty cut on axis '{h.axes[i].name}': [{lo}, {hi}] -> bins {a}..{b}")
        slices[i] = slice(a, b)
        applied[str(h.axes[i].name)] = (float(edges[a]), float(edges[b]))

    v, var = v[tuple(slices)], var[tuple(slices)]
    kept_edges = []
    for k in keep_idx:
        e = np.asarray(h.axes[k].edges, dtype=float)
        kept_edges.append(e if slices[k] == slice(None) else e[slices[k].start:slices[k].stop + 1])
    sum_axes = tuple(i for i in range(h.ndim) if i not in keep_idx)
    if sum_axes:
        v, var = v.sum(axis=sum_axes), var.sum(axis=sum_axes)
    remaining = sorted(keep_idx)
    perm = [remaining.index(k) for k in keep_idx]
    return np.transpose(v, perm), np.transpose(var, perm), kept_edges, applied, keep_idx


def _axis_title(ax):
    return str(getattr(ax, "label", "") or ax.name)


def makeSliceProjection(group, common_meta, output_path, samples, signal_label, keep, cuts, normalize, scale,
                        log_z, signal_scale, min_b_events, show_significance, plot_configuration=None):
    from .plots.annotations import addCMSBits
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration
    from matplotlib.colors import LogNorm

    pc = plot_configuration or PlotConfiguration()
    if len(keep) not in (1, 2):
        raise RuntimeError("keep must name 1 axis (1D plot) or 2 axes (2D plot)")
    by_label, unmatched = assign_samples(group, samples)
    if unmatched:
        print(f"[SliceProjection] ignoring datasets matching no sample pattern: {sorted(set(unmatched))}")
    present = [l for l in samples if by_label[l]]
    missing = [l for l in samples if not by_label[l]]
    if missing:
        print(f"[SliceProjection] no datasets found for sample(s) {missing}; plotting {present}")
    if not present:
        raise RuntimeError(f"no histogram matched any sample pattern {samples}; check meta['dataset_name'] and the "
                           f"grouping (group on pipeline + name, NOT dataset_name)")

    hists = {l: sum_histograms(by_label[l]) for l in present}
    ref = hists[present[0]]
    for l in present[1:]:
        if hists[l].ndim != ref.ndim or not all(np.array_equal(a.edges, b.edges) for a, b in zip(hists[l].axes, ref.axes)):
            raise RuntimeError(f"sample {l!r} has different axes than {present[0]!r}")

    data = {}
    for n_done, l in enumerate(present):
        v, var, kept_edges, applied, keep_idx = slice_project(hists[l], cuts, keep, warn=(n_done == 0))
        data[l] = (v, var)
    axes_kept = [ref.axes[k] for k in keep_idx]   # labels only; kept_edges holds the (possibly cut) bin edges
    cut_txt = "; ".join(f"{n} in [{lo:g}, {hi:g}]" for n, (lo, hi) in applied.items()) or "no cuts"

    # ---- yields: signal, each background, total background, S/sqrt(B) ----
    bkg_labels = [l for l in present if l != signal_label]
    sig_scale = {l: (signal_scale if l == signal_label else 1.0) for l in present}
    # yields and S/sqrt(B) use the REAL signal; signal_scale only multiplies what is drawn (so a x1000 curve is visible
    # next to the background without inflating the printed significance)
    tot = lambda l: (float(data[l][0].sum()), float(data[l][1].sum()))
    b_u = sum((float(data[l][0].sum()) for l in bkg_labels), 0.0)
    bv_u = sum((float(data[l][1].sum()) for l in bkg_labels), 0.0)
    print(f"\n=== SliceProjection [{common_meta.get('pipeline', '')}]  keep {[a.name for a in axes_kept]}  |  {cut_txt} ===")
    print(f"  {'sample':>10s} {'yield':>12s} {'+/-':>10s} {'n_eff':>9s}")
    rows = []
    for l in present:
        y, vy = tot(l)
        n_eff = _neff(data[l][0].sum(), data[l][1].sum())
        rows.append((l, y, float(np.sqrt(vy)), n_eff))
        print(f"  {l:>10s} {y:12.4g} {np.sqrt(vy):10.3g} {n_eff:9.1f}")
    if signal_label in present and bkg_labels:
        S, _ = tot(signal_label)
        n_b = _neff(b_u, bv_u)
        z, ze = _z(S, tot(signal_label)[1], b_u, bv_u, n_b, min_b_events)
        print(f"  {'bkg total':>10s} {b_u:12.4g} {np.sqrt(bv_u):10.3g} {n_b:9.1f}"
              f"\n  S/sqrt(B) = {('-' if not np.isfinite(z) else f'{z:.4f} +/- {ze:.4f}')}"
              f"   ('-' if n_eff(B) < {min_b_events:g})")

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{stem}_yields.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sample", "yield", "err", "n_eff", "cuts"])
        for l, y, e, n in rows:
            wr.writerow([l, y, e, n, cut_txt])

    colors = {l: SAMPLE_COLORS.get(l, f"C{i}") for i, l in enumerate(present)}
    title = f"{common_meta.get('pipeline', '')}   {cut_txt}"

    if len(keep) == 1:
        edges = kept_edges[0]
        fig, ax = plt.subplots()
        centres, widths = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
        ymin_pos = np.inf
        for l in present:
            v, var = data[l][0] * sig_scale[l], data[l][1] * sig_scale[l] ** 2
            if normalize:
                norm = v.sum() * widths if v.sum() > 0 else np.nan
                v, err = v / norm, np.sqrt(var) / norm
            else:
                err = np.sqrt(var)
            lab = l + (f" (x{signal_scale:g})" if l == signal_label and signal_scale != 1.0 else "")
            ax.stairs(v, edges, color=colors[l], label=lab, linewidth=1.8)
            ax.errorbar(centres, v, yerr=err, fmt="none", ecolor=colors[l], elinewidth=1.0, alpha=0.8)
            pos = v[np.isfinite(v) & (v > 0)]
            if pos.size:
                ymin_pos = min(ymin_pos, pos.min())
        ax.set_xlabel(_axis_title(axes_kept[0]))
        ax.set_ylabel("density" if normalize else "weighted events / bin")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_yscale(scale)
        if scale == "log" and np.isfinite(ymin_pos):
            ax.set_ylim(bottom=ymin_pos * 0.3, top=ax.get_ylim()[1] * 5)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(loc="upper right", frameon=False, title="\n".join(f"{n} in [{lo:g}, {hi:g}]" for n, (lo, hi) in applied.items()) or None,
                  title_fontsize=9, fontsize=11)
        addCMSBits(ax, [x.metadata for x in group], extra_text=f"{common_meta.get('pipeline', '')}", plot_configuration=pc)
        saveFig(fig, str(output_path), extension=pc.image_type)
        plt.close(fig)
        return

    # ---- 2D: one panel per sample (+ S/sqrt(B) per cell) ----
    xe, ye = kept_edges
    panels = [(l, data[l][0] * sig_scale[l], "viridis" if l == signal_label else "magma") for l in present]
    with np.errstate(divide="ignore", invalid="ignore"):
        if show_significance and signal_label in present and bkg_labels:
            B2 = sum(data[l][0] for l in bkg_labels)
            Bv2 = sum(data[l][1] for l in bkg_labels)
            neff_cell = np.where(Bv2 > 0, B2 ** 2 / Bv2, 0.0)
            zmap = np.where((neff_cell >= min_b_events) & (B2 > 0),
                            data[signal_label][0] / np.sqrt(np.where(B2 > 0, B2, np.nan)), np.nan)
            panels.append(("S/sqrt(B) per cell", zmap, "plasma"))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.4), squeeze=False, layout="constrained")
    for j, (ttl, arr, cmap) in enumerate(panels):
        ax = axes[0][j]
        arr = np.ma.masked_invalid(arr)
        if ttl != "S/sqrt(B) per cell":
            arr = np.ma.masked_where(arr <= 0, arr)
        arr = arr.T
        norm = None
        if log_z and ttl != "S/sqrt(B) per cell" and arr.count() and float(arr.max()) > float(arr.min()) > 0:
            norm = LogNorm(vmin=float(arr.min()), vmax=float(arr.max()))
        mesh = ax.pcolormesh(xe, ye, arr, cmap=cmap, norm=norm, shading="flat")
        fig.colorbar(mesh, ax=ax)
        if ttl == "S/sqrt(B) per cell" and not arr.count():
            ax.text(0.5, 0.5, f"no cell has n_eff(B) >= {min_b_events:g}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13)
        extra = ""
        if ttl in data:
            y, _ = tot(ttl)
            extra = f"\nyield = {y:.3g}"
        ax.set_title(ttl + extra, fontsize=15)
        ax.set_xlabel(_axis_title(axes_kept[0]))
        ax.set_ylabel(_axis_title(axes_kept[1]))
    fig.suptitle(title, fontsize=15)
    saveFig(fig, str(output_path), extension=pc.image_type)
    plt.close(fig)


@define
class SliceProjection(BasePostprocessor):
    """
    Cut and project an N-D histogram down to a 1D or 2D plot, one curve / panel per sample (signal, QCD, ttbar, ...).
    Made for the 4D fit histogram (m_Hbb, m_W1, m_HWW_threejet, score): put cuts on some axes, sum over the others,
    and look at what is left -- e.g. m_Hbb in a Won/HWW window with score >= 0.92, or the (m_W1, m_HWW) plane
    above a score cut. Also prints the yields, effective event counts and S/sqrt(B) of the selection and writes
    <output_name minus extension>_yields.csv. The same slice_project() function is what a datacard slicing step
    can reuse.

    Input: ONE N-D histogram per sample, all with identical axes. Group on {pipeline, name} -- NOT dataset_name -- so
    signal, QCD and ttbar land in one group; samples are then told apart by regex on meta["dataset_name"].
    Weights are used as OSCA made them (cross section x lumi already applied); several datasets of one sample are summed.

    samples      : {label: regex on dataset_name}, disjoint. e.g.
                   {"signal": "ggHH", "QCD": "(?i)qcd", "ttbar": "(?i)tt_(hadronic|semilep)"}
    signal_label : which label is the signal (the rest are backgrounds for S/sqrt(B)).
    keep         : 1 axis (1D plot) or 2 axes (2D, x then y), each an axis-name regex or an axis index.
    cuts         : {axis regex or index: [lo, hi]} value ranges kept on that axis, snapped to bin edges (a warning
                   is printed if a bound is not an edge). Use .inf / -.inf for an open end.
    normalize    : 1D only -- draw each sample as a density (compare shapes).
    scale        : 1D y axis, "linear" or "log".
    log_z        : 2D only -- logarithmic colour scale on the sample panels (default True).
    signal_scale : factor on the signal as DRAWN only (e.g. x1000 to make it visible in a 1D yield plot); the printed
                   yields, the CSV and S/sqrt(B) always use the real signal.
    min_b_events : S/sqrt(B) is only shown where the effective background event count is at least this.
    show_significance : 2D only -- add an S/sqrt(B) per-cell panel.
    """
    output_name: str
    samples: Dict[str, str]
    keep: List[str]
    signal_label: str = "signal"
    cuts: Dict[str, List[float]] = field(factory=dict)
    normalize: bool = False
    scale: str = "linear"
    log_z: bool = True
    signal_scale: float = 1.0
    min_b_events: float = 5.0
    show_significance: bool = True

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeSliceProjection, group, common_meta, output_path,
            samples=self.samples, signal_label=self.signal_label, keep=self.keep, cuts=self.cuts,
            normalize=self.normalize, scale=self.scale, log_z=self.log_z, signal_scale=self.signal_scale,
            min_b_events=self.min_b_events, show_significance=self.show_significance, plot_configuration=pc,
        )

# ----------------------------------------------------------------------------------------------
# plot 5: sensitivity ladder -- how much does each extra fit dimension add, and which score is best?
# ----------------------------------------------------------------------------------------------
def rebin_sum(a, axis, factor):
    """Merge `factor` neighbouring bins along `axis` by summing (the last group keeps the remainder)."""
    if factor <= 1:
        return a
    return np.add.reduceat(a, np.arange(0, a.shape[axis], factor), axis=axis)


def _cell_terms(S, vS, B, vB, min_b):
    """Per-cell usability (B > 0 and n_eff(B) >= min_b) and the cell's S^2/B and its MC-variance contribution."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Bs = np.where(B > 0, B, 1.0)
        neff = np.where(vB > 0, B ** 2 / np.where(vB > 0, vB, 1.0), 0.0)
        usable = (B > 0) & (neff >= min_b)
        z2 = np.where(usable, S ** 2 / Bs, 0.0)
        vz = np.where(usable, (2 * S / Bs) ** 2 * vS + (S ** 2 / Bs ** 2) ** 2 * vB, 0.0)
    return usable, z2, vz


def ladder_level(levels, min_b):
    """
    Significance Z = sqrt(sum over bins of S^2/B) of ONE ladder level, with ADAPTIVE refinement.
    levels = [(S, varS, B, varB) for ladder steps 0..K]; step k has the k+1 kept axes in ladder order, so the cells of
    step k+1 are the cells of step k times one new trailing axis. Refinement runs from the finest step K upward:
    a cell of step k is split into its step-k+1 children only if every resulting bin has n_eff(B) >= min_b
    (children that fail are merged into one remainder bin that must pass too); otherwise the cell stays whole.
    A cell that fails n_eff and cannot be split goes into one catch-all bin, which counts only if it passes itself.
    Splitting a bin can only raise sum S^2/B, so the ladder is not pushed DOWN by sparse QCD; extra dimensions only
    count where the QCD MC can support them. Returns (Z, error from the MC variances, signal fraction in counted
    bins, number of counted bins).
    """
    K = len(levels) - 1
    S, vS, B, vB = levels[K]
    usable, z2, vz = _cell_terms(S, vS, B, vB, min_b)
    st = {"z2": z2, "vz": vz, "sig": np.where(usable, S, 0.0), "n": usable.astype(float),
          "pS": np.where(usable, 0.0, S), "pvS": np.where(usable, 0.0, vS),
          "pB": np.where(usable, 0.0, B), "pvB": np.where(usable, 0.0, vB)}
    for k in range(K - 1, -1, -1):
        S, vS, B, vB = levels[k]
        cS, cvS, cB, cvB = levels[k + 1]
        u_ch, _, _ = _cell_terms(cS, cvS, cB, cvB, min_b)
        nu = ~u_ch
        rS, rvS, rB, rvB = ((a * nu).sum(axis=-1) for a in (cS, cvS, cB, cvB))
        r_use, rz2, rvz = _cell_terms(rS, rvS, rB, rvB, min_b)
        r_empty = (rS == 0) & (rB == 0)
        split = u_ch.any(axis=-1) & (r_empty | r_use)
        own_use, oz2, ovz = _cell_terms(S, vS, B, vB, min_b)
        sp = {"z2": (st["z2"] * u_ch).sum(axis=-1) + rz2, "vz": (st["vz"] * u_ch).sum(axis=-1) + rvz,
              "sig": (st["sig"] * u_ch).sum(axis=-1) + np.where(r_use, rS, 0.0),
              "n": (st["n"] * u_ch).sum(axis=-1) + r_use.astype(float),
              "pS": np.zeros_like(S), "pvS": np.zeros_like(S), "pB": np.zeros_like(S), "pvB": np.zeros_like(S)}
        un = {"z2": oz2, "vz": ovz, "sig": np.where(own_use, S, 0.0), "n": own_use.astype(float),
              "pS": np.where(own_use, 0.0, S), "pvS": np.where(own_use, 0.0, vS),
              "pB": np.where(own_use, 0.0, B), "pvB": np.where(own_use, 0.0, vB)}
        st = {key: np.where(split, sp[key], un[key]) for key in st}
    z2, vz, sig, n = float(st["z2"].sum()), float(st["vz"].sum()), float(st["sig"].sum()), float(st["n"].sum())
    pS, pvS, pB, pvB = (float(st[key].sum()) for key in ("pS", "pvS", "pB", "pvB"))
    p_use, pz2, pvz = _cell_terms(np.array(pS), np.array(pvS), np.array(pB), np.array(pvB), min_b)
    if bool(p_use):
        z2, vz, sig, n = z2 + float(pz2), vz + float(pvz), sig + pS, n + 1
    z = float(np.sqrt(z2))
    err = float(np.sqrt(vz) / (2 * z)) if z > 0 else float("nan")
    tot = float(levels[0][0].sum())
    return z, err, (sig / tot if tot > 0 else float("nan")), int(n)


def ladder_levels(vs, ws, vb, wb, keep_orders):
    """(S, varS, B, varB) for every ladder step: sum over the axes not kept, axes put in keep order."""
    out = []
    for keep in keep_orders:
        sum_axes = tuple(i for i in range(vs.ndim) if i not in keep)
        rem = sorted(keep)
        perm = [rem.index(a) for a in keep]
        out.append(tuple(np.transpose(a.sum(axis=sum_axes) if sum_axes else a, perm) for a in (vs, ws, vb, wb)))
    return out


def _arrays_nd(h):
    v = np.asarray(h.values(), dtype=float)
    var = h.variances()
    return v, (v.copy() if var is None else np.asarray(var, dtype=float))


def makeSensitivityLadder(group, common_meta, output_path, samples, signal_label, scores, name_key, order,
                          score_axis, mass_rebin, min_b_events, signal_scale, plot_configuration=None):
    from .plots.annotations import addCMSBits
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    available = sorted({str(x[1].get(name_key, "<missing>")) for x in group})
    results = {}                    # label -> {factor: [(z, err, frac, nbins) per step]}
    mass_width = None
    for label, name_regex in scores.items():
        chosen = [x for x in group if re.search(name_regex, str(x[1].get(name_key, ""))) is not None]
        if not chosen:
            raise RuntimeError(f"SensitivityLadder: no histogram matches {name_regex!r} (score {label!r}); "
                               f"histogram names present: {available}")
        by_label, _ = assign_samples(chosen, samples)
        bkg_labels = [l for l in samples if l != signal_label and by_label[l]]
        if not by_label.get(signal_label) or not bkg_labels:
            raise RuntimeError(
                f"SensitivityLadder: score {label!r} needs signal AND at least one background; found "
                f"{ {l: sorted({str(x[1].get('dataset_name')) for x in v}) for l, v in by_label.items()} }")
        hists = {l: sum_histograms(by_label[l]) for l in [signal_label] + bkg_labels}
        ref = hists[signal_label]
        idx = [resolve_axis(ref, k) for k in order]
        sidx = resolve_axis(ref, score_axis)
        if len({sidx, *idx}) != len(idx) + 1:
            raise RuntimeError(f"order {order} and score_axis {score_axis!r} must name distinct axes")
        arr = {l: _arrays_nd(h) for l, h in hists.items()}
        if mass_width is None:
            mass_width = float(np.mean(np.diff(np.asarray(ref.axes[idx[0]].edges, dtype=float))))
        results[label] = {}
        for f in mass_rebin:
            cur = {}
            for l, (v, var) in arr.items():
                for a in idx:
                    v, var = rebin_sum(v, a, f), rebin_sum(var, a, f)
                cur[l] = (v, var)
            vs, ws = cur[signal_label][0] * signal_scale, cur[signal_label][1] * signal_scale ** 2
            vb = sum(cur[l][0] for l in bkg_labels)
            wb = sum(cur[l][1] for l in bkg_labels)
            lv = ladder_levels(vs, ws, vb, wb, [[sidx] + idx[:k] for k in range(len(idx) + 1)])
            results[label][f] = [ladder_level(lv[:k + 1], min_b_events) for k in range(len(idx) + 1)]

    steps = ["score only"] + [f"+ {k}" for k in order]
    print(f"\n=== Sensitivity ladder [{common_meta.get('pipeline', '')}]: Z = sqrt(sum_bins S^2/B), min n_eff(B) {min_b_events:g}; "
          f"bins are only split where the QCD MC supports it ===")
    for f in mass_rebin:
        print(f"  mass bins x{f} ({mass_width * f:g} GeV)")
        print(f"    {'step':>12s}" + "".join(f" | {l:>26s}" for l in scores))
        for k, s in enumerate(steps):
            cells = []
            for l in scores:
                z, e, fr, nb = results[l][f][k]
                cells.append(f" | {z:8.4f} +/- {e:6.4f} ({100 * fr:3.0f}%, {nb:6d})")
            print(f"    {s:>12s}" + "".join(cells))
    print("  (in brackets: % of signal in bins that pass the n_eff test, then the number of bins; stat-only, binning tuned on the same MC -> optimistic)")

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{stem}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["score", "mass_rebin", "step", "Z", "Z_err", "signal_frac_usable_bins", "n_usable_bins"])
        for l in scores:
            for f in mass_rebin:
                for k, s in enumerate(steps):
                    z, e, fr, nb = results[l][f][k]
                    wr.writerow([l, f, s, z, e, fr, nb])

    n = len(mass_rebin)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 7), squeeze=False, layout="constrained", sharey=True)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    nl = len(scores)
    bw = 0.8 / nl
    for j, f in enumerate(mass_rebin):
        ax = axes[0][j]
        for i, l in enumerate(scores):
            xs = np.arange(len(steps)) + (i - (nl - 1) / 2) * bw
            z = np.array([p[0] for p in results[l][f]])
            e = np.nan_to_num(np.array([p[1] for p in results[l][f]]))
            ax.bar(xs, z, width=bw * 0.92, color=palette[i % len(palette)], label=l, yerr=e, ecolor="k", capsize=3)
            for x_, z_, p in zip(xs, z, results[l][f]):
                ax.text(x_, 0.01 * max(z.max(), 1e-12), f"{100 * p[2]:.0f}%", ha="center", va="bottom", fontsize=9,
                        rotation=90, color="w")
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels(steps, fontsize=14)
        ax.set_title(f"mass bins {mass_width * f:g} GeV", fontsize=16)
        if j == 0:
            ax.set_ylabel("sqrt(sum S^2/B)")
            ax.legend(loc="upper left", fontsize=13)
    ymax = max(p[0] + (p[1] if np.isfinite(p[1]) else 0) for l in scores for f in mass_rebin for p in results[l][f])
    axes[0][0].set_ylim(0, ymax * 1.25)
    addCMSBits(axes[0][0], [x.metadata for x in group], extra_text=f"{common_meta.get('pipeline', '')}", plot_configuration=pc)
    fig.suptitle(f"bar labels: % of signal in bins passing QCD n_eff >= {min_b_events:g}; stat-only, binning tuned on the same MC",
                 fontsize=11)
    saveFig(fig, str(output_path), extension=pc.image_type)
    plt.close(fig)


@define
class SensitivityLadder(BasePostprocessor):
    """
    How much does each extra fit dimension add, and which score is best? For each score (one 4D histogram per
    score: (m_Hbb, m_W1, m_HWW, score) or any order) it computes the cut-and-count significance
    Z = sqrt(sum over bins of S^2/B) with the score axis alone, then adding the mass axes one at a time
    (`order`), and draws grouped bars (one bar per score) with MC-variance error bars. `mass_rebin` repeats
    the ladder with the mass axes merged by those factors (1 = native bins, 5 = five bins merged), which shows how
    much of any gain is just fine binning of a thin QCD sample. Bins whose effective QCD count is below
    min_b_events are not dropped: a bin is split into finer bins only where every resulting bin passes that
    n_eff test (see ladder_level), so extra dimensions count only where the QCD MC supports them. The % printed on
    each bar is the share of the signal sitting in bins that pass the test.

    This is a STAT-ONLY, MC-tuned number: it is optimistic and does not replace a Combine fit; it ranks scores and
    shows what the extra dimensions buy.

    Input: the 4D histograms of all scores and samples; group on {pipeline, era.name} (NOT name or dataset_name);
    `scores` picks the histograms apart by regex on meta[name_key], `samples` picks the datasets apart by regex on
    meta["dataset_name"] (signal_label = the signal; every other present sample is background).
    Writes <output_name> and <output_name minus extension>.csv.

    scores       : {legend label: regex on the histogram name}
    order        : axis keys (name/label regex or index) added one by one after the score axis
    score_axis   : key of the score axis (a name regex that matches only that axis in every histogram, or an index)
    mass_rebin   : list of merge factors applied to every axis in `order`
    """
    output_name: str
    samples: Dict[str, str]
    scores: Dict[str, str]
    signal_label: str = "signal"
    name_key: str = "name"
    order: List[str] = field(factory=lambda: ["Hbb", "Won", "HWW"])
    score_axis: str = "score"
    mass_rebin: List[int] = field(factory=lambda: [1, 5])
    min_b_events: float = 5.0
    signal_scale: float = 1.0

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeSensitivityLadder, group, common_meta, output_path,
            samples=self.samples, signal_label=self.signal_label, scores=self.scores, name_key=self.name_key,
            order=self.order, score_axis=self.score_axis, mass_rebin=self.mass_rebin,
            min_b_events=self.min_b_events, signal_scale=self.signal_scale, plot_configuration=pc,
        )



# ----------------------------------------------------------------------------------------------
# plots 6 and 7: many mass-plane / mass-window variants of the 4D (m_Hbb, m_W1, m_HWW, score) fit histogram
# ----------------------------------------------------------------------------------------------
MASS_KEYS = ("Hbb", "Won", "HWW")
DEFAULT_AXES = {"Hbb": "Hbb", "Won": "Won", "HWW": "HWW", "score": "score"}
DEFAULT_CENTERS = {"Hbb": 125.2, "Won": 80.4, "HWW": 125.0}
DEFAULT_HALF_WIDTHS = {"Hbb": 20.0, "Won": 15.0, "HWW": 25.0}


def _tag(x):
    return "all" if x is None or not np.isfinite(x) else f"{x:g}".replace(".", "p").replace("-", "m")


def scan_inputs(group, samples, signal_label, scores, name_key, who, per_label=False):
    """{score label: (signal Hist, summed-background Hist, [background labels used])} from one group of 4D histograms.
    With per_label=True each value gets a 4th item: {background label: its own summed Hist}."""
    available = sorted({str(x[1].get(name_key, "<missing>")) for x in group})
    out = {}
    for label, name_regex in scores.items():
        chosen = [x for x in group if re.search(name_regex, str(x[1].get(name_key, ""))) is not None]
        if not chosen:
            raise RuntimeError(f"{who}: no histogram matches {name_regex!r} (score {label!r}); names present: {available}")
        by_label, _ = assign_samples(chosen, samples)
        bkg_labels = [l for l in samples if l != signal_label and by_label[l]]
        if not by_label.get(signal_label) or not bkg_labels:
            raise RuntimeError(f"{who}: score {label!r} needs signal AND at least one background; found "
                               f"{ {l: sorted({str(x[1].get('dataset_name')) for x in v}) for l, v in by_label.items()} }")
        hs = sum_histograms(by_label[signal_label])
        hb = sum_histograms([p for l in bkg_labels for p in by_label[l]])
        if hs.ndim != 4 or hb.ndim != 4:
            raise RuntimeError(f"{who}: expected 4D histograms, got {hs.ndim}D/{hb.ndim}D for score {label!r}")
        if per_label:
            out[label] = (hs, hb, bkg_labels, {l: sum_histograms(by_label[l]) for l in bkg_labels})
        else:
            out[label] = (hs, hb, bkg_labels)
    return out


def _slice_pair(hs, hb, cuts, keep):
    vs, ws, edges, _, _ = slice_project(hs, cuts, keep, warn=False)
    vb, wb, e2, _, _ = slice_project(hb, cuts, keep, warn=False)
    if not all(np.array_equal(a, b) for a, b in zip(edges, e2)):
        raise RuntimeError("signal and background histograms have different axes")
    return vs, ws, vb, wb, edges


def makeMassPlaneScan(group, common_meta, output_path, samples, signal_label, scores, name_key, axes, planes,
                      third_windows, det_bin_edges, centers, half_widths, signal_scale, bkg_scale, min_b_events,
                      background_label, window_shape, top_n, plot_configuration=None, n_parts=1,
                      hbb_column=True, rows_per_figure=0, display_signal_scale=1000.0, proj_scale="log",
                      separate_backgrounds=True):
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    inputs = scan_inputs(group, samples, signal_label, scores, name_key, "MassPlaneScan", per_label=True)
    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, (hs, hb, bkg_labels, hb_by) in inputs.items():
        bg_name = " + ".join(bkg_labels) if background_label == "auto" else background_label
        aidx = {k: resolve_axis(hs, v) for k, v in axes.items()}
        if len(set(aidx.values())) != 4:
            raise RuntimeError(f"MassPlaneScan: axes {axes} do not resolve to four distinct axes: {aidx}")
        for xk, yk in planes:
            thirds = [k for k in MASS_KEYS if k not in (xk, yk)]
            if len(thirds) != 1 or xk == yk:
                raise RuntimeError(f"MassPlaneScan: a plane must be two different mass axes from {MASS_KEYS}, got {(xk, yk)}")
            third = thirds[0]
            for win in third_windows.get(third, [[None, None]]):
                lo, hi = (list(win) + [None, None])[:2]
                lo = None if (lo is None or not np.isfinite(lo)) else float(lo)    # yaml uses -.inf / .inf for an open end
                hi = None if (hi is None or not np.isfinite(hi)) else float(hi)
                cuts = {} if (lo is None and hi is None) else {axes[third]: [lo, hi]}
                vs, ws, vb, wb, edges = _slice_pair(hs, hb, cuts, [axes[xk], axes[yk], axes["score"]])
                plane_parts = None
                if separate_backgrounds and len(hb_by) > 1:
                    plane_parts = {}
                    for bl, hbl in hb_by.items():
                        pv, pw, _, _, _ = slice_project(hbl, cuts, [axes[xk], axes[yk], axes["score"]], warn=False)
                        plane_parts[bl] = (pv, pw)
                wtxt = f"{third} in [{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]" if cuts else f"{third} summed over"
                tag = f"{label}_{xk}-{yk}_{third}_{_tag(lo)}_{_tag(hi)}".replace(" ", "_")
                third_arg = None
                if hbb_column:
                    # box cuts = the dashed window on the plane (bounding box if the window is an ellipse); keep third + score
                    pcuts = {axes[xk]: [centers[xk] - half_widths[xk], centers[xk] + half_widths[xk]],
                             axes[yk]: [centers[yk] - half_widths[yk], centers[yk] + half_widths[yk]]}
                    pvs, pws, pvb, pwb, pedges = _slice_pair(hs, hb, pcuts, [axes[third], axes["score"]])
                    tparts = None
                    if separate_backgrounds and len(hb_by) > 1:
                        tparts = {}
                        for bl, hbl in hb_by.items():
                            tv, tw, _, _, _ = slice_project(hbl, pcuts, [axes[third], axes["score"]], warn=False)
                            tparts[bl] = (tv, tw)
                    third_arg = {"vs": pvs, "ws": pws, "vb": pvb, "wb": pwb, "edges": pedges[0], "label": third,
                                 "parts": tparts,
                                 "center": centers[third], "hw": half_widths[third], "display_scale": display_signal_scale,
                                 "scale": proj_scale,
                                 "cuts_txt": f"{xk} {pcuts[axes[xk]][0]:g}-{pcuts[axes[xk]][1]:g}, "
                                             f"{yk} {pcuts[axes[yk]][0]:g}-{pcuts[axes[yk]][1]:g} GeV"}
                s = makeDetectionBinned2D(
                    group, common_meta, f"{stem}_{tag}{ext}", "", "", 0, 1, 2, det_bin_edges, signal_scale, bkg_scale,
                    min_b_events, centers[xk], centers[yk], half_widths[xk], half_widths[yk], plot_configuration=pc,
                    score_label=label, window_shape=window_shape, arrays=(vs, ws, vb, wb, edges),
                    x_label=f"m({xk}) [GeV]", y_label=f"m({yk}) [GeV]", background_label=bg_name,
                    title_tag=f"[{xk} x {yk}; {wtxt}]", make_summary_plot=False, n_parts=n_parts,
                    third=third_arg, rows_per_figure=rows_per_figure, bkg_parts=plane_parts)
                rows.append({"score": label, "plane": f"{xk} x {yk}", "third_axis": third, "third_lo": lo, "third_hi": hi,
                             "tag": tag, **s})

    cols = ["score", "plane", "third_axis", "third_lo", "third_hi", "z_total", "n_cells", "best_cut", "best_z_window",
            "S_total", "B_total", "tag"]
    with open(f"{stem}_scan_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in cols})
    ranked = sorted((r for r in rows if np.isfinite(r["z_total"])), key=lambda r: -r["z_total"])[:top_n]
    print(f"\n=== MassPlaneScan: {len(rows)} variants; top {len(ranked)} by sqrt(sum S^2/B) over score bins and unmasked cells "
          f"(stat-only; the plane and windows were chosen on the same MC) ===")
    for r in ranked:
        w = "" if r["third_lo"] is None and r["third_hi"] is None else f" [{r['third_lo']}, {r['third_hi']}]"
        print(f"  {r['z_total']:8.4f}  {r['score']:>14s}  {r['plane']:>10s}  {r['third_axis']}{w if w else ' summed'}  cells {r['n_cells']}")
    if ranked:
        fig, ax = plt.subplots(figsize=(13, 0.5 * len(ranked) + 2.5), layout="constrained")
        labels = [f"{r['score']} | {r['plane']} | {r['third_axis']}"
                  + ("" if r["third_lo"] is None and r["third_hi"] is None else f" [{r['third_lo']:g},{r['third_hi']:g}]")
                  for r in ranked]
        ax.barh(range(len(ranked)), [r["z_total"] for r in ranked], color="#2980b9")
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("sqrt(sum S^2/B) over score bins and cells (stat-only)")
        ax.set_title(f"{common_meta.get('pipeline', '')}  top mass-plane variants", fontsize=14)
        saveFig(fig, f"{stem}_ranking{ext}", extension=pc.image_type)
        plt.close(fig)


@define
class MassPlaneScan(BasePostprocessor):
    """
    Many versions of the detection-score-binned mass-plane study, from the 4D fit histogram
    (m_Hbb, m_W1, m_HWW_threejet, score). For every score histogram, every mass PLANE in `planes` (x, y) and every
    window on the remaining third mass axis in `third_windows`, it writes the same grid DetectionBinned2D makes
    (rows = score ranges; signal | background | S/sqrt(B) per cell), then ranks all variants.

    Example: planes [[Won, HWW], [Hbb, HWW], [Hbb, Won]] and third_windows {Hbb: [[-.inf,.inf],[105,145]],
    Won: [[-.inf,.inf],[65,95]], HWW: [[-.inf,.inf],[90,160]]} gives Won x HWW with Hbb summed and with Hbb in
    105-145, and so on. Each variant writes <stem>_<score>_<plane>_<third>_<lo>_<hi>.<ext>; the whole scan adds
    <stem>_scan_summary.csv and <stem>_ranking.<ext>.

    Input: the 4D histograms of all scores and samples; group on {pipeline, era.name} (NOT name or dataset_name), same
    as SensitivityLadder. The numbers are STAT-ONLY and MC-tuned (optimistic); with many variants, the best of them
    is also biased upward, so use the ranking to choose what to try next, not as a measured sensitivity.

    samples, signal_label, scores, name_key : as SensitivityLadder.
    axes          : {Hbb, Won, HWW, score: axis name/label regex or index}.
    planes        : [[x, y], ...] with x, y from Hbb / Won / HWW.
    third_windows : {mass key: [[lo, hi], ...]} windows on the axis that is NOT in the plane (-.inf / .inf = open end in yaml).
    det_bin_edges : score ranges (rows); snapped to the score-axis edges (0.04 steps in the 4D histogram).
    centers, half_widths : window used for the in-window numbers on each plane axis (box or ellipse).
    background_label : name for the background panels; "auto" builds it from the background samples that are present
                    (e.g. "QCD" now, "QCD + ttbar" once ttbar is in the group).
    n_parts       : split each grid (rows = score ranges) into this many figures (<stem>_..._part1, _part2, ...), as
                    DetectionBinned2D does; default 1 = one figure per variant.
    rows_per_figure : if > 0, overrides n_parts: that many score-range rows per figure (2 = two rows per image).
    hbb_column    : add a 4th column to every row: the projection of the third mass axis (Hbb) for that score range,
                    after cutting the plane's two axes to their window box (the dashed box: centers +/- half_widths;
                    a bounding box if window_shape is ellipse). Signal x display_signal_scale vs background, grey band =
                    the third axis's own window, in-window S/sqrt(B) printed. proj_scale: "log" or "linear".
    separate_backgrounds : when two or more background samples are present (QCD and ttbar), draw one 2D panel per
                    background plus the S/sqrt(B) panel from their total, and one curve per background (plus the total,
                    dashed) in the Hbb column. False = a single summed background panel as before.
    top_n         : how many variants the ranking plot shows.
    """
    output_name: str
    samples: Dict[str, str]
    scores: Dict[str, str]
    signal_label: str = "signal"
    name_key: str = "name"
    axes: Dict[str, str] = field(factory=lambda: dict(DEFAULT_AXES))
    planes: List[List[str]] = field(factory=lambda: [["Won", "HWW"], ["Hbb", "HWW"], ["Hbb", "Won"]])
    third_windows: Dict[str, List[List[float]]] = field(factory=dict)
    det_bin_edges: List[float] = field(factory=lambda: [0.0, 0.5, 0.8, 0.92, 0.96, 1.0])
    centers: Dict[str, float] = field(factory=lambda: dict(DEFAULT_CENTERS))
    half_widths: Dict[str, float] = field(factory=lambda: dict(DEFAULT_HALF_WIDTHS))
    signal_scale: float = 1.0
    bkg_scale: float = 1.0
    min_b_events: float = 5.0
    background_label: str = "QCD"
    window_shape: str = "box"
    top_n: int = 25
    n_parts: int = 1
    rows_per_figure: int = 0
    hbb_column: bool = True
    display_signal_scale: float = 1000.0
    proj_scale: str = "log"
    separate_backgrounds: bool = True

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeMassPlaneScan, group, common_meta, output_path,
            samples=self.samples, signal_label=self.signal_label, scores=self.scores, name_key=self.name_key,
            axes={**DEFAULT_AXES, **self.axes}, planes=self.planes, third_windows=self.third_windows,
            det_bin_edges=self.det_bin_edges, centers={**DEFAULT_CENTERS, **self.centers},
            half_widths={**DEFAULT_HALF_WIDTHS, **self.half_widths}, signal_scale=self.signal_scale,
            bkg_scale=self.bkg_scale, min_b_events=self.min_b_events, background_label=self.background_label,
            window_shape=self.window_shape, top_n=self.top_n, plot_configuration=pc, n_parts=self.n_parts,
            hbb_column=self.hbb_column, rows_per_figure=self.rows_per_figure,
            display_signal_scale=self.display_signal_scale, proj_scale=self.proj_scale,
            separate_backgrounds=self.separate_backgrounds,
        )


def makeMassProjectionScan(group, common_meta, output_path, samples, signal_label, scores, name_key, axes, project,
                           scan_axes, centers, scan_half_widths, extra_windows, score_cuts, signal_scale,
                           display_signal_scale, min_b_events, background_label, scale, rows_per_figure, top_n,
                           project_half_width, plot_configuration=None):
    import itertools
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    if project not in MASS_KEYS or any(k not in MASS_KEYS or k == project for k in scan_axes):
        raise RuntimeError(f"MassProjectionScan: project and scan_axes must be different keys from {MASS_KEYS}")
    inputs = scan_inputs(group, samples, signal_label, scores, name_key, "MassProjectionScan")

    # windows: unrestricted baseline, then the product of half-widths over the scan axes, then explicit extras
    windows = [("no window", {})]
    for hws in itertools.product(*[scan_half_widths.get(k, [DEFAULT_HALF_WIDTHS[k]]) for k in scan_axes]):
        win = {k: [centers[k] - hw, centers[k] + hw] for k, hw in zip(scan_axes, hws)}
        windows.append((", ".join(f"{k} {centers[k]:g}+/-{hw:g}" for k, hw in zip(scan_axes, hws)), win))
    for win in extra_windows:
        windows.append((", ".join(f"{k} [{b[0]:g},{b[1]:g}]" for k, b in win.items()), dict(win)))

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    pc_center, proj_hw = centers[project], project_half_width
    records = []
    for label, (hs, hb, bkg_labels) in inputs.items():
        aidx = {k: resolve_axis(hs, v) for k, v in axes.items()}
        if len(set(aidx.values())) != 4:
            raise RuntimeError(f"MassProjectionScan: axes {axes} do not resolve to four distinct axes: {aidx}")
        panels = {}
        for wi, (wname, win) in enumerate(windows):
            for cut in score_cuts:
                cuts = {axes[k]: b for k, b in win.items()}
                cuts[axes["score"]] = [cut, None]
                vs, ws, vb, wb, edges = _slice_pair(hs, hb, cuts, [axes[project]])
                S, vS = vs * signal_scale, ws * signal_scale ** 2
                usable, z2, _ = _cell_terms(S, vS, vb, wb, min_b_events)
                z_quad = float(np.sqrt(z2.sum())) if usable.any() else float("nan")
                e = edges[0]
                c = 0.5 * (e[:-1] + e[1:])
                inw = np.abs(c - pc_center) <= proj_hw
                Sw, Bw, vSw, vBw = S[inw].sum(), vb[inw].sum(), vS[inw].sum(), wb[inw].sum()
                z_win, z_win_err = _z(float(Sw), float(vSw), float(Bw), float(vBw), _neff(Bw, vBw), min_b_events)
                rec = {"score": label, "window": wname, "score_cut": cut, "S": float(S.sum()), "B": float(vb.sum()),
                       "neff_B": _neff(vb.sum(), wb.sum()), "z_quad": z_quad, "z_proj_window": z_win,
                       "z_proj_window_err": z_win_err}
                records.append(rec)
                panels[(wi, cut)] = (edges[0], S, vS, vb, wb, rec)

        # figures: rows = windows, columns = score cuts
        for part, start in enumerate(range(0, len(windows), rows_per_figure)):
            idx = list(range(start, min(len(windows), start + rows_per_figure)))
            fig, axs = plt.subplots(len(idx), len(score_cuts), figsize=(6.4 * len(score_cuts), 4.2 * len(idx)),
                                    squeeze=False, layout="constrained", sharex=True)
            for r, wi in enumerate(idx):
                for cj, cut in enumerate(score_cuts):
                    ax = axs[r][cj]
                    e, S, vS, vb, wb, rec = panels[(wi, cut)]
                    ax.stairs(vb, e, color=SAMPLE_COLORS["QCD"], linewidth=1.6, label=background_label)
                    ax.stairs(S * display_signal_scale, e, color=SAMPLE_COLORS["signal"], linewidth=1.6,
                              label="signal" + (f" x{display_signal_scale:g}" if display_signal_scale != 1 else ""))
                    ax.axvspan(pc_center - proj_hw, pc_center + proj_hw, color="0.85", alpha=0.5, zorder=0)
                    ax.set_yscale(scale)
                    if scale == "log":
                        pos = np.concatenate([vb[vb > 0], (S * display_signal_scale)[S > 0]])
                        if pos.size:
                            ax.set_ylim(pos.min() * 0.5, pos.max() * 8)
                    fz = lambda v: "-" if not np.isfinite(v) else f"{v:.3f}"
                    ax.set_title(f"{windows[wi][0]}\nscore >= {cut:g}", fontsize=11)
                    ax.text(0.97, 0.95, f"S/sqrt(B): bins {fz(rec['z_quad'])}\nin {project} window {fz(rec['z_proj_window'])}\n"
                            f"S={rec['S']:.3g}  n_eff(B)={rec['neff_B']:.0f}", transform=ax.transAxes, ha="right", va="top",
                            fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.85))
                    if r == len(idx) - 1:
                        ax.set_xlabel(f"m({project}) [GeV]")
                    if cj == 0:
                        ax.set_ylabel("weighted events / bin")
                    if r == 0 and cj == 0:
                        ax.legend(loc="lower left", fontsize=9, frameon=False)
            fig.suptitle(f"{common_meta.get('pipeline', '')}  {label}: m({project}) in windows on {' and '.join(scan_axes)} "
                         f"(grey band = {project} window; stat-only, MC-tuned)", fontsize=13)
            out = f"{stem}_{label}_part{part + 1}{ext}".replace(" ", "_")
            saveFig(fig, out, extension=pc.image_type)
            plt.close(fig)

    cols = ["score", "window", "score_cut", "S", "B", "neff_B", "z_quad", "z_proj_window", "z_proj_window_err"]
    with open(f"{stem}_projection_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r_ in records:
            wr.writerow({k: r_[k] for k in cols})
    ranked = sorted((r_ for r_ in records if np.isfinite(r_["z_quad"])), key=lambda r_: -r_["z_quad"])[:top_n]
    print(f"\n=== MassProjectionScan: {len(records)} (window, score cut) combinations; top {len(ranked)} by "
          f"sqrt(sum_bins S^2/B) of m({project}) (stat-only, MC-tuned) ===")
    for r_ in ranked:
        print(f"  {r_['z_quad']:8.4f}  {r_['score']:>14s}  score>={r_['score_cut']:g}  {r_['window']}  "
              f"[in {project} window: {r_['z_proj_window']:.3f}]")


@define
class MassProjectionScan(BasePostprocessor):
    """
    m_Hbb (or any one mass axis) in MANY windows on the other two masses, from the 4D fit histogram. For each score
    histogram it builds a grid with one row per window and one column per score cut, each panel showing the projected
    mass of signal and background with S/sqrt(B) (quadrature over the projected bins, and inside the projected
    axis's own mass window). Windows are: no window, then every combination of the half-widths in `scan_half_widths`
    around `centers` on the `scan_axes`, then any `extra_windows`. Writes <stem>_<score>_partN.<ext>,
    <stem>_projection_summary.csv and prints the best combinations.

    Same input rules as MassPlaneScan (4D histograms; group on {pipeline, era.name}). Stat-only and MC-tuned: many
    windows and cuts are being tried on one MC sample, so the best entry is biased upward.

    project          : mass key shown on the x axis (default Hbb).
    scan_axes        : mass keys the windows are placed on (default [Won, HWW]).
    centers          : window centres in GeV (defaults Hbb 125.2, Won 80.4, HWW 125).
    project_half_width : half-width (GeV) of the projected axis's own window, used for the "in window" number.
    scan_half_widths : {mass key: [half-widths in GeV]} -- all combinations across the scan axes are drawn.
    extra_windows    : [{mass key: [lo, hi], ...}, ...] explicit windows to add.
    score_cuts       : keep score >= each value (snapped to the score-axis edges, 0.04 steps in the 4D histogram).
    signal_scale     : real factor on the signal in the statistics (default 1).
    display_signal_scale : factor applied to the signal curve only, so it is visible next to the background.
    scale            : "log" (default) or "linear" y axis.
    rows_per_figure  : windows per figure; more windows spill into further parts.
    """
    output_name: str
    samples: Dict[str, str]
    scores: Dict[str, str]
    signal_label: str = "signal"
    name_key: str = "name"
    axes: Dict[str, str] = field(factory=lambda: dict(DEFAULT_AXES))
    project: str = "Hbb"
    scan_axes: List[str] = field(factory=lambda: ["Won", "HWW"])
    centers: Dict[str, float] = field(factory=lambda: dict(DEFAULT_CENTERS))
    scan_half_widths: Dict[str, List[float]] = field(factory=lambda: {"Won": [10.0, 15.0, 25.0], "HWW": [20.0, 30.0, 50.0]})
    extra_windows: List[Dict[str, List[float]]] = field(factory=list)
    project_half_width: float = 20.0
    score_cuts: List[float] = field(factory=lambda: [0.8, 0.92, 0.96])
    signal_scale: float = 1.0
    display_signal_scale: float = 1000.0
    min_b_events: float = 5.0
    background_label: str = "QCD"
    scale: str = "log"
    rows_per_figure: int = 5
    top_n: int = 25

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeMassProjectionScan, group, common_meta, output_path,
            samples=self.samples, signal_label=self.signal_label, scores=self.scores, name_key=self.name_key,
            axes={**DEFAULT_AXES, **self.axes}, project=self.project, scan_axes=self.scan_axes,
            centers={**DEFAULT_CENTERS, **self.centers}, scan_half_widths=self.scan_half_widths,
            extra_windows=self.extra_windows, score_cuts=self.score_cuts, signal_scale=self.signal_scale,
            display_signal_scale=self.display_signal_scale, min_b_events=self.min_b_events,
            background_label=self.background_label, scale=self.scale, rows_per_figure=self.rows_per_figure,
            top_n=self.top_n, project_half_width=self.project_half_width, plot_configuration=pc,
        )

# ----------------------------------------------------------------------------------------------
# plot 8: a COLUMN of m_Hbb projections, one row per score range, with fixed windows on m_Won and m_HWW
# ----------------------------------------------------------------------------------------------
def makeMassProjectionByScore(group, common_meta, output_path, samples, signal_label, scores, name_key, axes, project,
                              cuts, score_edges, centers, project_half_width, signal_scale, display_signal_scale,
                              normalize, min_b_events, scale, rows_per_figure, plot_configuration=None):
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    if project not in MASS_KEYS or project in cuts or any(k not in MASS_KEYS for k in cuts):
        raise RuntimeError(f"MassProjectionByScore: project {project!r} must be one of {MASS_KEYS} and cuts must be on the "
                           f"OTHER mass keys, got cuts on {sorted(cuts)}")
    edges_s = sorted(set(float(x) for x in score_edges))
    if len(edges_s) < 2:
        raise RuntimeError("MassProjectionByScore: score_edges needs at least two values")
    ranges = list(zip(edges_s[:-1], edges_s[1:]))
    available = sorted({str(x[1].get(name_key, "<missing>")) for x in group})
    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    palette = {"signal": SAMPLE_COLORS["signal"], "QCD": SAMPLE_COLORS["QCD"], "ttbar": SAMPLE_COLORS["ttbar"]}
    mass_cuts = {}
    for k, b in cuts.items():
        lo, hi = (list(b) + [None, None])[:2]
        mass_cuts[axes[k]] = [None if (lo is None or not np.isfinite(lo)) else float(lo),
                              None if (hi is None or not np.isfinite(hi)) else float(hi)]
    cut_txt = ", ".join(f"{k} in [{b[0]:g}, {b[1]:g}]" for k, b in cuts.items())
    summary_rows = []

    for label, name_regex in scores.items():
        chosen = [x for x in group if re.search(name_regex, str(x[1].get(name_key, ""))) is not None]
        if not chosen:
            raise RuntimeError(f"MassProjectionByScore: no histogram matches {name_regex!r} (score {label!r}); "
                               f"names present: {available}")
        by_label, _ = assign_samples(chosen, samples)
        bkg_labels = [l for l in samples if l != signal_label and by_label[l]]
        if not by_label.get(signal_label) or not bkg_labels:
            raise RuntimeError(f"MassProjectionByScore: score {label!r} needs signal AND at least one background")
        hists = {l: sum_histograms(by_label[l]) for l in [signal_label] + bkg_labels}
        ref = hists[signal_label]
        if ref.ndim != 4:
            raise RuntimeError(f"MassProjectionByScore: expected 4D histograms, got {ref.ndim}D for score {label!r}")
        aidx = {k: resolve_axis(ref, v) for k, v in axes.items()}
        if len(set(aidx.values())) != 4:
            raise RuntimeError(f"MassProjectionByScore: axes {axes} do not resolve to four distinct axes: {aidx}")

        panels = []
        for lo, hi in ranges:
            c = dict(mass_cuts)
            c[axes["score"]] = [lo, hi]
            proj = {}
            for l, h in hists.items():
                v, w, e, _, _ = slice_project(h, c, [axes[project]], warn=False)
                proj[l] = (v, w)
            edges = e[0]
            S, vS = proj[signal_label][0] * signal_scale, proj[signal_label][1] * signal_scale ** 2
            B = sum(proj[l][0] for l in bkg_labels)
            vB = sum(proj[l][1] for l in bkg_labels)
            usable, z2, _ = _cell_terms(S, vS, B, vB, min_b_events)
            z_quad = float(np.sqrt(z2.sum())) if usable.any() else float("nan")
            ctr = 0.5 * (edges[:-1] + edges[1:])
            inw = np.abs(ctr - centers[project]) <= project_half_width
            z_win, z_win_err = _z(float(S[inw].sum()), float(vS[inw].sum()), float(B[inw].sum()), float(vB[inw].sum()),
                                  _neff(B[inw].sum(), vB[inw].sum()), min_b_events)
            rec = {"score": label, "lo": lo, "hi": hi, "S": float(S.sum()), "B": float(B.sum()),
                   "neff_B": _neff(B.sum(), vB.sum()), "z_quad": z_quad, "z_proj_window": z_win,
                   "z_proj_window_err": z_win_err}
            summary_rows.append(rec)
            panels.append((edges, proj, S, rec))

        # ---- figure(s): ONE column, one row per score range ----
        for part, start in enumerate(range(0, len(panels), rows_per_figure)):
            idx = list(range(start, min(len(panels), start + rows_per_figure)))
            fig, axs = plt.subplots(len(idx), 1, figsize=(9.0, 3.4 * len(idx)), squeeze=False, layout="constrained",
                                    sharex=True)
            for r, i in enumerate(idx):
                ax = axs[r][0]
                edges, proj, S, rec = panels[i]
                lo, hi = ranges[i]
                curves = [(signal_label, S, SAMPLE_COLORS["signal"])] + \
                         [(l, proj[l][0], palette.get(l, "0.4")) for l in bkg_labels]
                allpos = []
                for name, y, col in curves:
                    y = np.asarray(y, dtype=float)
                    if name == signal_label and not normalize and display_signal_scale != 1:
                        y = y * display_signal_scale
                        name = f"{name} x{display_signal_scale:g}"
                    if normalize:
                        tot = y.sum()
                        y = y / tot if tot > 0 else y
                    ax.stairs(y, edges, color=col, linewidth=1.6, label=name)
                    allpos.append(y[y > 0])
                ax.axvspan(centers[project] - project_half_width, centers[project] + project_half_width,
                           color="0.85", alpha=0.5, zorder=0)
                ax.set_yscale(scale)
                pos = np.concatenate(allpos) if allpos else np.array([])
                if scale == "log" and pos.size:
                    ax.set_ylim(pos.min() * 0.5, pos.max() * 30)
                elif pos.size:
                    ax.set_ylim(0, pos.max() * 1.6)
                fz = lambda v: "-" if not np.isfinite(v) else f"{v:.3f}"
                ax.set_title(f"{label} score in [{lo:.2f}, {hi:.2f}{']' if hi == edges_s[-1] else ')'}", fontsize=13)
                ax.text(0.97, 0.93, f"S/sqrt(B): bins {fz(rec['z_quad'])}, in {project} window {fz(rec['z_proj_window'])}\n"
                        f"S={rec['S']:.3g}  B={rec['B']:.3g}  n_eff(B)={rec['neff_B']:.0f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.85))
                ax.set_ylabel("fraction of events / bin" if normalize else "weighted events / bin")
                if r == 0:
                    ax.legend(loc="upper left", fontsize=10, frameon=False)
                if r == len(idx) - 1:
                    ax.set_xlabel(f"m({project}) [GeV]")
            fig.suptitle(f"{common_meta.get('pipeline', '')}: m({project}) by {label}-score range\n{cut_txt} "
                         f"(part {part + 1}/{-(-len(panels) // rows_per_figure)}; grey band = {project} window; "
                         f"stat-only)", fontsize=11)
            saveFig(fig, f"{stem}_{label}_part{part + 1}{ext}".replace(" ", "_"), extension=pc.image_type)
            plt.close(fig)

    cols = ["score", "lo", "hi", "S", "B", "neff_B", "z_quad", "z_proj_window", "z_proj_window_err"]
    with open(f"{stem}_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r_ in summary_rows:
            wr.writerow({k: r_[k] for k in cols})
    print(f"\n=== MassProjectionByScore: m({project}) with {cut_txt}; rows = score ranges "
          f"(S/sqrt(B) stat-only; '-' = fewer than {min_b_events:g} effective background events) ===")
    for r_ in summary_rows:
        print(f"  {r_['score']:>14s} [{r_['lo']:.2f},{r_['hi']:.2f})  S={r_['S']:9.3g}  B={r_['B']:9.3g}  "
              f"n_eff(B)={r_['neff_B']:7.1f}  bins {('-' if not np.isfinite(r_['z_quad']) else f'{r_['z_quad']:.4f}'):>8s}  "
              f"in-window {('-' if not np.isfinite(r_['z_proj_window']) else f'{r_['z_proj_window']:.4f}'):>8s}")


@define
class MassProjectionByScore(BasePostprocessor):
    """
    A COLUMN of m_Hbb (or any one mass axis) projections after the multi-distribution selection: windows on the other
    two masses (cuts) and one ROW per detection-score range. Each panel overlays signal and every background that is
    present (QCD, and ttbar once it is in the group) and prints S/sqrt(B) over the projected bins and inside the
    projected axis's own mass window. One figure per score histogram (split into parts of rows_per_figure rows).

    Input: the 4D histograms of all scores and samples; group on {pipeline, era.name} (NOT name or dataset_name), same
    as MassPlaneScan / SensitivityLadder. Writes <stem>_<score>_part<N>.<ext> and <stem>_summary.csv, prints the table.

    samples, signal_label, scores, name_key, axes : as MassPlaneScan.
    project            : the mass axis that is drawn (Hbb / Won / HWW).
    cuts               : {mass key: [lo, hi]} windows on the other mass axes (-.inf / .inf = open end).
    score_edges        : score-range edges; each consecutive pair is a row. The 4D score axis has edges every 0.04.
    normalize          : True = each curve scaled to unit area (shape comparison); False = weighted yields, the signal
                         multiplied by display_signal_scale so it is visible next to the background.
    scale              : "log" or "linear" y axis.
    rows_per_figure    : score ranges per figure.
    """
    output_name: str
    samples: Dict[str, str]
    scores: Dict[str, str]
    signal_label: str = "signal"
    name_key: str = "name"
    axes: Dict[str, str] = field(factory=lambda: dict(DEFAULT_AXES))
    project: str = "Hbb"
    cuts: Dict[str, List[float]] = field(factory=lambda: {"Won": [60.0, 100.0], "HWW": [40.0, 120.0]})
    score_edges: List[float] = field(factory=lambda: [0.80, 0.84, 0.88, 0.92, 0.96, 1.0])
    centers: Dict[str, float] = field(factory=lambda: dict(DEFAULT_CENTERS))
    project_half_width: float = 20.0
    signal_scale: float = 1.0
    display_signal_scale: float = 1000.0
    normalize: bool = False
    min_b_events: float = 5.0
    scale: str = "log"
    rows_per_figure: int = 6

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeMassProjectionByScore, group, common_meta, output_path,
            samples=self.samples, signal_label=self.signal_label, scores=self.scores, name_key=self.name_key,
            axes={**DEFAULT_AXES, **self.axes}, project=self.project, cuts=self.cuts, score_edges=self.score_edges,
            centers={**DEFAULT_CENTERS, **self.centers}, project_half_width=self.project_half_width,
            signal_scale=self.signal_scale, display_signal_scale=self.display_signal_scale, normalize=self.normalize,
            min_b_events=self.min_b_events, scale=self.scale, rows_per_figure=self.rows_per_figure, plot_configuration=pc,
        )