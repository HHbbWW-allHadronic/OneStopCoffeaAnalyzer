"""OSCA postprocessor: sideband Gaussian-process background estimate of the m_bb window.

UNTESTED against OSCA (written from the patterns in detection_plots.py); the numerics live in GP_core.py,
which is tested on toys. Put both files in the postprocessing package next to detection_plots.py.

Input: one 1D histogram of the mass variable (e.g. b_dijet_mass) per dataset. Bin edges must land on the
SR/CR boundaries (the code raises otherwise). Group WITHOUT dataset_name (same rule as
DetectionCutMassProjection) so every background dataset of a pipeline lands in one group.
"""
from __future__ import annotations

import csv
import functools as ft
import re
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from attrs import define, field

from analyzer.utils.structure_tools import commonDict, dictToDot, dotFormat
from .processors import BasePostprocessor
from .detection_plots import sum_histograms, _hist1d_arrays, split_ext
from .GP_core import predict_window, poly_ftest


def _neff(values, variances):
    """Effective number of MC events per bin, (sum w)^2 / sum w^2 (nan where the bin is empty)."""
    v = np.asarray(variances, float)
    return np.where(v > 0, np.asarray(values, float) ** 2 / np.where(v > 0, v, 1.0), np.nan)


def _base_axes(r, values, variances, sr, crs, mass_label, plot_range=None):
    """Shared frame: whole histogram range, filled points = control-region and window bins, open grey points =
    bins outside both (not fitted), dashed lines = control-region edges, grey box = signal region. The legend sits
    to the right of the axes; the caller adds its curves and calls _finish_legend."""
    c = r["centers"]
    w = float(c[1] - c[0])
    x_lo, x_hi = (c[0] - w / 2, c[-1] + w / 2) if plot_range is None else tuple(plot_range)
    show = (c >= x_lo) & (c <= x_hi)
    used = r["cr_mask"] | r["sr_mask"]
    err_mc = np.sqrt(variances)
    # layout="none": the CMS plot style can switch on a layout engine, and that makes subplots_adjust a no-op
    try:
        fig, ax = plt.subplots(figsize=(16, 9.5), layout="none")
    except TypeError:                      # matplotlib < 3.5 has no layout= argument
        fig, ax = plt.subplots(figsize=(16, 9.5), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.64, top=0.93, bottom=0.11)   # right margin holds the legend
    ax.axvspan(sr[0], sr[1], color="0.9", zorder=0, label="signal region")
    m_used, m_out = show & used, show & ~used
    ax.errorbar(c[m_used], values[m_used], err_mc[m_used], fmt="ko", ms=5, label="background MC", zorder=3)
    if m_out.any():
        ax.errorbar(c[m_out], values[m_out], err_mc[m_out], fmt="o", color="0.55", mfc="none", ms=5,
                    label="outside regions (not fitted)", zorder=3)
    for a_, b_ in crs:
        for xe in (a_, b_):
            ax.axvline(xe, color="0.35", ls="--", lw=1.4, zorder=1)
    top = np.nanmax((values + err_mc)[m_used]) if m_used.any() else 1.0     # y scale from the fitted bins only;
    bot = np.nanmin((values - err_mc)[m_used]) if m_used.any() else 0.0     # taller outside bins run off the top
    lower = max(bot - 0.1 * (top - bot), 0.0)
    ax.set_ylim(lower, top + 0.5 * (top - lower))                           # headroom for the CMS label
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(mass_label)
    ax.set_ylabel("weighted events / bin")
    return fig, ax


def _finish_legend(ax, text_lines):
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="0.35", ls="--", lw=1.4)); l.append("control region edges")
    h += [Line2D([], [], linestyle="none", marker="none")] * len(text_lines)
    l += list(text_lines)
    ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=False, fontsize=14,
              handlelength=1.8, borderaxespad=0.0)


def _draw_gp_plot(r, values, variances, sr, crs, mass_label, plot_range=None):
    """GP prediction in the window against the MC."""
    fig, ax = _base_axes(r, values, variances, sr, crs, mass_label, plot_range)
    sc = r["centers"][r["sr_mask"]]
    # band = total prediction error (GP + jackknife) applied as a relative error, fully correlated across bins
    err = np.abs(r["bin_mean"]) * (r["pred_err"] / r["pred"] if r["pred"] else 0.0)
    ax.fill_between(sc, r["bin_mean"] - err, r["bin_mean"] + err, color="C1", alpha=0.4, label="prediction uncertainty")
    ax.plot(sc, r["bin_mean"], color="C1", lw=2, label="GP prediction", zorder=4)
    _finish_legend(ax, [f"SR predicted: {r['pred']:.3g} +/- {r['pred_err']:.2g}",
                        f"SR MC: {r['obs']:.3g} +/- {r['obs_err']:.2g}", f"pull: {r['pull']:+.1f}"])
    return fig, ax


def _draw_fit_plot(r, values, variances, sr, crs, mass_label, ftr, plot_range=None):
    """Simple sideband fits (ax+b, the F-test-selected polynomial, exponential) extrapolated through the window."""
    fig, ax = _base_axes(r, values, variances, sr, crs, mass_label, plot_range)
    xs = np.linspace(min(a_ for a_, _ in crs), max(b_ for _, b_ in crs), 200)
    d = ftr["selected_degree"]
    rows = {q["label"]: q for q in ftr["rows"]}
    lines = []
    lin = rows.get("poly deg 1 (ax+b)")
    if 1 in ftr["curves"]:
        ax.plot(xs, ftr["curves"][1](xs), color="C2", ls=":", lw=2.2, label="linear fit (ax+b)", zorder=2)
        if lin:
            lines.append(f"ax+b: {lin['pred']:.3g}, pull {lin['pull']:+.1f}")
    if d != 1:
        sel = [q for q in ftr["rows"] if q["selected"]][0]
        ax.plot(xs, ftr["curves"][d](xs), color="C0", ls="-.", lw=2.2, label=f"polynomial deg {d} (F-test)", zorder=2)
        lines.append(f"poly deg {d}: {sel['pred']:.3g}, pull {sel['pull']:+.1f}")
    if ftr["exp_curve"] is not None and "exponential" in rows:
        ax.plot(xs, ftr["exp_curve"](xs), color="C4", ls="--", lw=2.0, label="exponential", zorder=2)
        q = rows["exponential"]
        lines.append(f"exponential: {q['pred']:.3g}, pull {q['pull']:+.1f}")
    sel = [q for q in ftr["rows"] if q["selected"]][0]
    head = [f"SR MC: {r['obs']:.3g} +/- {r['obs_err']:.2g}",
            f"F-test selects degree {d}" + (f" (p={sel['F_p']:.2g} vs {d - 1})" if sel["F_p"] == sel["F_p"] else "")]
    _finish_legend(ax, head + ["SR predicted:"] + lines)
    return fig, ax


def makeGPBackgroundEstimate(group, common_meta, output_path, background_regex, sr, crs, plot_configuration=None,
                             mass_label="m_bb [GeV]", score_cut=None, mass_axis=0, smooth_variance=True, mean_model="exp",
                             ftest_max_degree=4, plot_range=None):
    from .plots.annotations import addCMSBits
    from .plots.utils import saveFig
    from .plots.common import PlotConfiguration

    pc = plot_configuration or PlotConfiguration()
    bkg = [x for x in group if re.search(background_regex, str(x[1].get("dataset_name", ""))) is not None]
    if not bkg:
        print(f"[GPBackgroundEstimate] skipping {output_path}: no dataset matches {background_regex!r} "
              f"[{common_meta.get('pipeline', '')}]")
        return None
    h = sum_histograms(bkg)
    cut_txt = ""
    if h.ndim == 1:
        values, variances, edges = _hist1d_arrays(h)
    elif h.ndim == 2:
        # mass x score histogram: keep score bins whose lower edge is >= score_cut, project onto the mass axis
        if score_cut is None:
            raise RuntimeError("2D histogram given but score_cut is not set")
        m_ax, s_ax = mass_axis, 1 - mass_axis
        v2 = np.asarray(h.values(), dtype=float)
        w2 = h.variances()
        w2 = v2.copy() if w2 is None else np.asarray(w2, dtype=float)
        s_edges = np.asarray(h.axes[s_ax].edges, dtype=float)
        i = int(np.argmin(np.abs(s_edges - score_cut)))
        if abs(s_edges[i] - score_cut) > 1e-6:
            print(f"[GPBackgroundEstimate] score_cut {score_cut:g} is not a score-axis edge; using {s_edges[i]:g}")
        sl = [slice(None), slice(None)]
        sl[s_ax] = slice(i, None)
        values = v2[tuple(sl)].sum(axis=s_ax)
        variances = w2[tuple(sl)].sum(axis=s_ax)
        edges = np.asarray(h.axes[m_ax].edges, dtype=float)
        cut_txt = f" score >= {s_edges[i]:g}"
    else:
        raise RuntimeError(f"expected a 1D mass histogram or a 2D (mass, score) one, got {h.ndim}D "
                           f"({[a.name for a in h.axes]})")
    r = predict_window(edges, values, variances, tuple(sr), [tuple(c) for c in crs], smooth_var=smooth_variance, mean=mean_model)

    print(f"\n=== GP sideband estimate [{common_meta.get('pipeline', '')}]{cut_txt}: SR {sr[0]:g}-{sr[1]:g}, "
          f"CR {', '.join(f'{a:g}-{b:g}' for a, b in crs)} ===\n"
          f"  predicted SR yield = {r['pred']:.4g} +/- {r['pred_err']:.3g}   "
          f"MC SR yield = {r['obs']:.4g} +/- {r['obs_err']:.3g}   pull = {r['pull']:+.2f}   "
          f"(sigma={r['sigma']:.3g}, ell={r['ell']:.3g} GeV)\n"
          f"  prediction error: GP {r['pred_err_gp']:.3g}, jackknife over CR bins {r['pred_err_jk']:.3g} "
          f"(total {r['pred_err']:.3g} = {r['pred_err'] / r['pred']:.1%})")
    ne = _neff(values, variances)
    print(f"  effective MC events per bin: CR median {np.nanmedian(ne[r['cr_mask']]):.1f} "
          f"(min {np.nanmin(ne[r['cr_mask']]):.1f}), SR median {np.nanmedian(ne[r['sr_mask']]):.1f}"
          f"   [smooth_variance={smooth_variance}, mean_model={mean_model}]")

    ftr = None
    if ftest_max_degree is not None and ftest_max_degree >= 0:
        try:
            ftr = poly_ftest(edges, values, variances, tuple(sr), [tuple(c_) for c_ in crs],
                             max_degree=ftest_max_degree, smooth_var=smooth_variance)
            print(f"  simple sideband fits + F-test ({ftr['n_cr']} control bins, alpha={ftr['alpha']:g}); "
                  f"MC window yield {ftr['obs']:.4g}")
            print(f"    {'fit':18s} {'npar':>4s} {'chi2/ndf':>9s} {'F':>9s} {'p(F)':>8s} {'SR pred':>11s} "
                  f"{'+/-':>9s} {'pull':>6s}")
            for q in ftr["rows"]:
                print(f"    {q['label']:18s} {q['npar']:4d} {q['chi2_ndf']:9.2f} {q['F']:9.2f} {q['F_p']:8.3f} "
                      f"{q['pred']:11.4g} {q['pred_err']:9.3g} {q['pull']:+6.2f}"
                      f"{'   <== selected by F-test' if q['selected'] else ''}")
        except Exception as e:                      # never let the extra fits kill the main plot
            print(f"  [GPBackgroundEstimate] polynomial/F-test fits skipped: {e!r}")

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{stem}_gp.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bin_center", "region", "mc", "mc_err", "neff", "gp_mean", "gp_err"])
        c, sm, cm = r["centers"], r["sr_mask"], r["cr_mask"]
        k = 0
        for i in range(len(c)):
            reg = "SR" if sm[i] else ("CR" if cm[i] else "other")
            gm = ge = ""
            if sm[i]:
                gm, ge = r["bin_mean"][k], np.sqrt(max(r["bin_cov"][k, k], 0.0)); k += 1
            wr.writerow([c[i], reg, values[i], np.sqrt(variances[i]), _neff(values, variances)[i], gm, ge])

    if ftr is not None:
        with open(f"{stem}_ftest.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["fit", "npar", "chi2", "ndf", "chi2_ndf", "gof_p", "F", "F_p", "sr_pred", "sr_pred_err",
                         "sr_mc", "pull", "selected"])
            for q in ftr["rows"]:
                wr.writerow([q["label"], q["npar"], q["chi2"], q["ndf"], q["chi2_ndf"], q["gof_p"], q["F"], q["F_p"],
                             q["pred"], q["pred_err"], ftr["obs"], q["pull"], int(q["selected"])])

    meta = [x.metadata for x in group]
    extra_text = f"{common_meta.get('pipeline', '')}{cut_txt}"
    fig, ax = _draw_gp_plot(r, values, variances, sr, crs, mass_label, plot_range=plot_range)
    addCMSBits(ax, meta, extra_text=extra_text, plot_configuration=pc)
    saveFig(fig, str(output_path), extension=pc.image_type)
    plt.close(fig)
    if ftr is not None:                                   # separate plot for the simple fits
        fig, ax = _draw_fit_plot(r, values, variances, sr, crs, mass_label, ftr, plot_range=plot_range)
        addCMSBits(ax, meta, extra_text=extra_text, plot_configuration=pc)
        saveFig(fig, f"{stem}_fits{ext}", extension=pc.image_type)
        plt.close(fig)
    return r


@define
class GPBackgroundEstimate(BasePostprocessor):
    """
    Fit a Gaussian process to the control-region bins of a 1D mass histogram (background MC) and predict the
    blinded signal-region yield; the plot and <output_name minus extension>_gp.csv hold prediction vs MC.

    background_regex : regex on meta["dataset_name"] picking the background datasets to sum.
    sr               : [lo, hi] signal region.
    crs              : list of [lo, hi] control regions (sidebands).
    """
    output_name: str
    background_regex: str
    sr: List[float] = field(factory=lambda: [100.0, 150.0])
    crs: List[List[float]] = field(factory=lambda: [[50.0, 100.0], [150.0, 200.0]])
    mass_label: str = "m_bb [GeV]"
    score_cut: float | None = None   # only for a 2D (mass, score) histogram: keep score >= score_cut
    mass_axis: int = 0               # which axis of the 2D histogram is the mass (the other is the score)
    mean_model: str = "exp"          # prior mean for the GP: "exp" (exponential falloff) or "linear" (straight line)
    smooth_variance: bool = True     # smooth the control-bin variances (avoids a low bias when MC weights are large)
    ftest_max_degree: int = 4        # simple polynomial fits (deg 0..N) of the control bins + F-test; -1 turns it off
    plot_range: List[float] | None = None   # x range of the plot; default = the whole histogram

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeGPBackgroundEstimate, group, common_meta, output_path,
            background_regex=self.background_regex, sr=self.sr, crs=self.crs,
            plot_configuration=pc, mass_label=self.mass_label,
            score_cut=self.score_cut, mass_axis=self.mass_axis, smooth_variance=self.smooth_variance, mean_model=self.mean_model,
            ftest_max_degree=self.ftest_max_degree, plot_range=self.plot_range,
        )