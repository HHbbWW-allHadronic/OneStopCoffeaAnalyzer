"""OSCA postprocessor: sideband Gaussian-process background estimate of the m_bb window.

UNTESTED against OSCA (written from the patterns in detection_plots.py); the numerics live in gp_core.py,
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
from .gp_core import predict_window


def makeGPBackgroundEstimate(group, common_meta, output_path, background_regex, sr, crs, plot_configuration=None,
                             mass_label="m_bb [GeV]", score_cut=None, mass_axis=0):
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
    r = predict_window(edges, values, variances, tuple(sr), [tuple(c) for c in crs])

    print(f"\n=== GP sideband estimate [{common_meta.get('pipeline', '')}]{cut_txt}: SR {sr[0]:g}-{sr[1]:g}, "
          f"CR {', '.join(f'{a:g}-{b:g}' for a, b in crs)} ===\n"
          f"  predicted SR yield = {r['pred']:.4g} +/- {r['pred_err']:.3g}   "
          f"MC SR yield = {r['obs']:.4g} +/- {r['obs_err']:.3g}   pull = {r['pull']:+.2f}   "
          f"(sigma={r['sigma']:.3g}, ell={r['ell']:.3g} GeV)")

    stem, ext = split_ext(output_path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{stem}_gp.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bin_center", "region", "mc", "mc_err", "gp_mean", "gp_err"])
        c, sm, cm = r["centers"], r["sr_mask"], r["cr_mask"]
        k = 0
        for i in range(len(c)):
            reg = "SR" if sm[i] else ("CR" if cm[i] else "other")
            gm = ge = ""
            if sm[i]:
                gm, ge = r["bin_mean"][k], np.sqrt(max(r["bin_cov"][k, k], 0.0)); k += 1
            wr.writerow([c[i], reg, values[i], np.sqrt(variances[i]), gm, ge])

    lo, hi = min(a for a, _ in crs), max(b for _, b in crs)
    c = r["centers"]
    show = (c >= lo) & (c < hi)
    fig, ax = plt.subplots()
    ax.errorbar(c[show], values[show], np.sqrt(variances[show]), fmt="ko", label="background MC")
    ax.axvspan(sr[0], sr[1], color="0.9", zorder=0, label="signal region")
    sc = c[r["sr_mask"]]
    err = np.sqrt(np.clip(np.diag(r["bin_cov"]), 0, None))
    ax.fill_between(sc, r["bin_mean"] - err, r["bin_mean"] + err, color="C1", alpha=0.4)
    ax.plot(sc, r["bin_mean"], color="C1", label="GP prediction")
    ax.set_xlabel(mass_label)
    ax.set_ylabel("weighted events / bin")
    ax.text(0.97, 0.95, f"SR: pred {r['pred']:.3g} +/- {r['pred_err']:.2g}\nMC {r['obs']:.3g}, pull {r['pull']:+.1f}",
            transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="upper center", frameon=False)
    addCMSBits(ax, [x.metadata for x in group], extra_text=f"{common_meta.get('pipeline', '')}{cut_txt}",
               plot_configuration=pc)
    saveFig(fig, str(output_path), extension=pc.image_type)
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

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(self.output_name, **dict(dictToDot(common_meta)), prefix=prefix)
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            makeGPBackgroundEstimate, group, common_meta, output_path,
            background_regex=self.background_regex, sr=self.sr, crs=self.crs,
            plot_configuration=pc, mass_label=self.mass_label,
            score_cut=self.score_cut, mass_axis=self.mass_axis,
        )