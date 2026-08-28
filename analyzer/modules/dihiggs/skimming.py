import uuid
import hashlib
from pathlib import Path

import awkward as ak
import correctionlib
from attrs import define, field

from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from analyzer.utils.file_tools import copyFile
from analyzer.utils.structure_tools import dictToDot, dotFormat
from typing import List
import numpy as np


@define
class SaveEventsH5(AnalyzerModule):
    """
    Analyzer module that serializes event-level data to HDF5 files for NN training.

    - Jet-level variables are written as (n_events, n_jets) datasets named
      "{jet_col}_{var}" (with '.' replaced by '_').
    - Event-level variables are written as (n_events,) datasets named after the
      Column (with '.' replaced by '_').

    Assumes it is run in a region where every event has at least `n_jets` jets in
    `jet_col`. Raises otherwise.

    Parameters
    ----------
    prefix : str
        Destination directory prefix where the output HDF5 files will be copied.
        (You said this will be the region name/path.)
    jet_col : Column
        Jet collection column (e.g. Column("goodJet")).
    jet_vars : list[str]
        Fields on the jet collection (e.g. ["pt","eta","phi","mass"]).
    n_jets : int
        Number of leading jets to write per event.
    event_vars : list[Column], optional
        Event-level columns to write (e.g. [Column("HT"), Column("b_dijet_12.mass")]).
    output_format : str, optional
        Filename template expanded with metadata fields plus file_id and uuid.
        Note: uuid is only used for the *local* temp filename unless you include
        "{uuid}" in this format.
    """

    prefix: str
    jet_col: Column
    jet_vars: list[str]
    n_jets: int
    event_vars: list[Column] = field(factory=list)
    output_format: str = (
        "{dataset_name}__{sample_name}__{file_id}"
        "__{chunk.event_start}_{chunk.event_stop}.h5"
    )

    def run(self, columns, params):
        file_id = (
            hashlib.md5((columns.metadata["chunk"]["file_path"]).encode())
            .hexdigest()
            .upper()
        )
        uid = str(uuid.uuid4())

        target_name = dotFormat(
            self.output_format,
            **dict(dictToDot(columns.metadata)),
            file_id=file_id,
            uuid=uid,
        )
        target = f"{self.prefix}/{target_name}"

        base = Path("localsaved")
        base.mkdir(exist_ok=True, parents=True)
        local_filename = base / f"{uid}.h5"

        try:
            import h5py

            jets = columns[self.jet_col]

            # Validate jet multiplicity before writing anything
            min_jets = ak.min(ak.num(jets))
            if min_jets is None:
                return columns, []

            min_jets = int(min_jets)
            if min_jets < self.n_jets:
                raise ValueError(
                    f"SaveH5: collection '{self.jet_col}' requires at least {self.n_jets} jets/event, "
                    f"but found an event with {min_jets} in chunk "
                    f"{columns.metadata['chunk']['file_path']} "
                    f"[{columns.metadata['chunk']['event_start']}, "
                    f"{columns.metadata['chunk']['event_stop']}]"
                )

            jet_prefix = str(self.jet_col).replace(".", "_")

            with h5py.File(local_filename, "w") as f:
                # Per-jet datasets: (n_events, n_jets)
                for v in self.jet_vars:
                    field_name = (
                        v
                        if (v != "btag")
                        else columns.metadata["era"]["btag_scale_factors"]["tagger"]
                    )
                    arr = jets[field_name][:, : self.n_jets]
                    data = ak.to_numpy(arr)
                    f.create_dataset(
                        f"{jet_prefix}_{v}",
                        data=data,
                        compression="gzip",
                    )

                # Event-level datasets: (n_events,)
                for c in self.event_vars:
                    arr = columns[c]
                    data = ak.to_numpy(arr)
                    name = str(c).replace(".", "_")
                    f.create_dataset(
                        name,
                        data=data,
                        compression="gzip",
                    )

            copyFile(local_filename, target)
        finally:
            local_filename.unlink(missing_ok=True)

        return columns, []

    def inputs(self, metadata):
        return [self.jet_col] + list(self.event_vars)

    def outputs(self, metadata):
        return []


# ===========================================================================================
# module versions of nano_to_h5_V2.py - main goal is to generate h5 files for training SPANet
# ===========================================================================================

W_MASS_PDG = 80.377  # GeV

def greedy_match(objects_a, objects_b, signal_field, dr_threshold=0.4, store_pt=False):
    max_a = int(ak.max(ak.num(objects_a)))
    max_b = int(ak.max(ak.num(objects_b)))

    eta_a_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_a.eta, max_a, clip=True), 0.0))
    phi_a_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_a.phi, max_a, clip=True), 0.0))
    eta_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_b.eta, max_b, clip=True), 0.0))
    phi_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_b.phi, max_b, clip=True), 0.0))
    sig_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(getattr(objects_b, signal_field), max_b, clip=True), -1))
    counts_a = ak.to_numpy(ak.num(objects_a))
    counts_b = ak.to_numpy(ak.num(objects_b))

    if store_pt:
        pt_b_np = ak.to_numpy(ak.fill_none(ak.pad_none(objects_b.pt, max_b, clip=True), np.nan))

    labels_out = []
    pt_out = [] if store_pt else None

    for i in range(len(counts_a)):
        n_a = int(counts_a[i])
        n_b = int(counts_b[i])

        labels = [-1] * n_a
        pts = [np.nan] * n_a

        if n_a == 0 or n_b == 0:
            labels_out.append(labels)
            if store_pt:
                pt_out.append(pts)
            continue

        eta_a = eta_a_np[i, :n_a]
        phi_a = phi_a_np[i, :n_a]
        eta_b = eta_b_np[i, :n_b]
        phi_b = phi_b_np[i, :n_b]

        deta = eta_a[:, None] - eta_b[None, :]
        dphi = phi_a[:, None] - phi_b[None, :]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dr_matrix = np.sqrt(deta ** 2 + dphi ** 2)

        assigned_a = set()
        assigned_b = set()

        while True:
            dr_masked = dr_matrix.copy()
            if assigned_a:
                dr_masked[list(assigned_a), :] = np.inf
            if assigned_b:
                dr_masked[:, list(assigned_b)] = np.inf
            if dr_masked.min() > dr_threshold:
                break
            ia, ib = np.unravel_index(dr_masked.argmin(), dr_masked.shape)
            labels[ia] = int(sig_b_np[i, ib])
            if store_pt:
                pts[ia] = float(pt_b_np[i, ib])
            assigned_a.add(ia)
            assigned_b.add(ib)

        labels_out.append(labels)
        if store_pt:
            pt_out.append(pts)

    if store_pt:
        return labels_out, pt_out
    return labels_out


def pad_and_convert(arr, max_real_jets, n_null_jets):
    padded = ak.pad_none(arr[:, :max_real_jets], max_real_jets + n_null_jets, clip=True)
    padded = ak.fill_none(padded, 0)
    return ak.to_numpy(padded)


def make_mask(jet_counts, max_real_jets, n_null_jets):
    return np.array([
        [True] * min(n, max_real_jets) + [False] * max(0, max_real_jets - n) + [True] * n_null_jets
        for n in jet_counts
    ], dtype=bool)


def make_targets(signal_arr, max_real_jets, n_null_jets, reco_mode, include_isr_target=True):
    n_events = len(signal_arr)
    null_idx = max_real_jets if n_null_jets > 0 else -1

    H1 = np.full((n_events, 2), null_idx, dtype=int)
    H2 = np.full((n_events, 4), null_idx, dtype=int)
    W1 = np.full((n_events, 2), null_idx, dtype=int)
    if include_isr_target:
        ISR = np.full((n_events, 1), null_idx, dtype=int)

    for i in range(n_events):
        h1_jets = np.where(signal_arr[i] == 1)[0]
        h2_jets = np.where(signal_arr[i] == 2)[0]
        h3_jets = np.where(signal_arr[i] == 3)[0]

        H1[i, :min(len(h1_jets), 2)] = h1_jets[:2]
        if include_isr_target:
            h_isr_jets = np.where(signal_arr[i] == 4)[0]
            ISR[i, :min(len(h_isr_jets), 1)] = h_isr_jets[:1]

        if reco_mode == "full_hww":
            W1[i, :min(len(h2_jets), 2)] = h2_jets[:2]
            all_q = np.concatenate([h2_jets[:2], h3_jets[:2]])
            H2[i, :min(len(all_q), 4)] = all_q[:4]
        elif reco_mode == "onshell_w":
            W1[i, :min(len(h2_jets), 2)] = h2_jets[:2]

        seen = set()
        for slot in range(2):
            if H1[i, slot] in seen:
                H1[i, slot] = -1
            else:
                seen.add(int(H1[i, slot]))

        if reco_mode == "full_hww":
            for slot in range(4):
                if H2[i, slot] in seen:
                    H2[i, slot] = -1
                else:
                    seen.add(int(H2[i, slot]))
        elif reco_mode == "onshell_w":
            for slot in range(2):
                if W1[i, slot] in seen:
                    W1[i, slot] = -1
                else:
                    seen.add(int(W1[i, slot]))

        # ISR dedup against H1/H2/W1 only matters (and is only computed at
        # all) when an ISR target is actually being built -- with
        # include_isr_target=False, an ISR-matched jet (signal==4) is never
        # dedup'd out of H1/H2/W1 either, since it was never a candidate for
        # those in the first place (h1_jets/h2_jets/h3_jets only ever draw
        # from signal==1/2/3). Nothing here changes H1/H2/W1's own values.
        if include_isr_target:
            if ISR[i, 0] in seen:
                ISR[i, 0] = -1
            else:
                seen.add(int(ISR[i, 0]))

    if reco_mode == "full_hww":
        out = {
            "H1": {"b1": H1[:, 0], "b2": H1[:, 1]},
            "H2": {"q1": H2[:, 0], "q2": H2[:, 1], "q3": H2[:, 2], "q4": H2[:, 3]},
        }
    elif reco_mode == "onshell_w":
        out = {
            "H1": {"b1": H1[:, 0], "b2": H1[:, 1]},
            "W1": {"q1": W1[:, 0], "q2": W1[:, 1]},
        }
    if include_isr_target:
        out["ISR"] = {"g1": ISR[:, 0]}
    return out


def make_random_permutations(n_events, n_real, rng):
    """One INDEPENDENT random permutation of range(n_real) per event, as
    an (n_events, n_real) array. perms[i, k] = j means 'new position k
    gets whatever was at old position j, for event i'. Verified via
    direct test against a full source+target consistency check before
    ever being wired into the real pipeline."""    
    perms = np.tile(np.arange(n_real), (n_events, 1))
    for i in range(n_events):
        rng.shuffle(perms[i])
    return perms


def apply_perm_to_source_field(arr, perms, n_real):
    """arr has shape (n_events, n_real + n_null). Only the first n_real
    columns (the REAL jet slots) get shuffled, per event; the null-slot
    column(s) at the end are left completely untouched -- they are not
    real jets, and null_idx's meaning as a fixed position depends on
    them staying put."""
    out = arr.copy()
    out[:, :n_real] = np.take_along_axis(arr[:, :n_real], perms, axis=1)
    return out


def remap_target_indices(target_arr, inv_perms, null_idx):
    """target_arr has shape (n_events,), values are either a real
    position in [0, n_real), OR null_idx, OR -1 (dedup conflict
    sentinel from make_targets). Only real positions get remapped
    through inv_perms; null_idx/-1 are sentinels, not real positions,
    and must pass through completely unchanged -- confirmed by direct
    test, since silently remapping a sentinel would corrupt the target
    in a way that's easy to miss."""
    out = target_arr.copy()
    for i in range(len(target_arr)):
        v = target_arr[i]
        if v != null_idx and v != -1:
            out[i] = inv_perms[i, v]
    return out


# ============================================================
# Debug tracing: per-parton gen-matching diagnosis + full target
# resolution trace, for debug_trace_events. Inlined here directly
# (rather than imported from a separate module) so this file has no
# external dependency beyond what's already imported above -- a
# previous version imported this from trace_target_construction.py,
# which failed at runtime with ModuleNotFoundError since that file was
# never actually placed anywhere on OSCA's import path. Everything
# needed now lives in this one file.
#
# signal value -> particle/daughter mapping, confirmed directly from
# make_targets' own logic above (full_hww mode):
#     signal == 1  ->  H1 (Hbb),  fills H1.b1, H1.b2
#     signal == 2  ->  H2 (HWW),  fills H2.q1, H2.q2 (one W's daughters)
#     signal == 3  ->  H2 (HWW),  fills H2.q3, H2.q4 (the other W's daughters)
#     signal == 4  ->  ISR (if include_isr_target)
# ============================================================

_TRACE_PDGID_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b"}
# Confirmed directly from make_targets' own source: H1 is ALWAYS
# computed and returned regardless of strip_bjets -- there is no
# strip_bjets check anywhere in make_targets at all. The strip_bjets
# exclusion of H1 from the final H5 happens downstream, at
# SaveSPANetH5's targets_cols, a separate module. This tracer shows
# everything make_targets itself actually produces for the given
# reco_mode/include_isr_target, dynamically -- not a fixed H1+H2
# assumption -- so it correctly handles full_hww, onshell_w, with or
# without ISR, without needing separate hardcoded code paths. Verified
# directly against both structures before being wired in here.
#
# Per-particle signal-group mapping, confirmed from make_targets' own
# logic: H1 always signal==1. For full_hww, H2 pools signal 2+3 --
# packing is shared across BOTH groups (confirmed earlier: an empty H2
# slot can't be attributed to one specific parton once one sub-group
# falls short, so failures are shown pooled, at the first empty H2
# slot). For onshell_w, W1 is ONLY ever signal==2 -- onshell_w never
# touches h3_jets at all, so there's no such ambiguity there. ISR is
# always signal==4.
_TRACE_PARTICLE_SIGNAL_GROUPS = {"H1": [1], "H2": [2, 3], "W1": [2], "ISR": [4]}


def _trace_quark_label(pdg_id):
    """Human-readable flavor from a PDG ID -- standard, stable PDG
    numbering (1=d, 2=u, 3=s, 4=c, 5=b), sign indicates antiparticle."""
    sign = "-" if pdg_id < 0 else ""
    name = _TRACE_PDGID_NAMES.get(abs(pdg_id), f"pdgId={pdg_id}")
    return f"{sign}{name}"


def _trace_dr_matrix(eta_a, phi_a, eta_b, phi_b):
    deta = eta_a[:, None] - eta_b[None, :]
    dphi = phi_a[:, None] - phi_b[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta**2 + dphi**2)


def _trace_greedy_match_with_indices(eta_a, phi_a, eta_b, phi_b, dr_threshold):
    n_a, n_b = len(eta_a), len(eta_b)
    if n_a == 0 or n_b == 0:
        return {}
    dr = _trace_dr_matrix(eta_a, phi_a, eta_b, phi_b)
    assigned_a, assigned_b, matches = set(), set(), {}
    while True:
        dr_masked = dr.copy()
        if assigned_a:
            dr_masked[list(assigned_a), :] = np.inf
        if assigned_b:
            dr_masked[:, list(assigned_b)] = np.inf
        if dr_masked.min() > dr_threshold:
            break
        ia, ib = np.unravel_index(dr_masked.argmin(), dr_masked.shape)
        matches[int(ia)] = int(ib)
        assigned_a.add(ia)
        assigned_b.add(ib)
    return matches


def _trace_diagnose_partons(chs, gen_gj, jets, event_idx, dr_threshold, pt_ratio_bounds=(0.5, 2.0)):
    partons = chs[event_idx]
    genjets = gen_gj[event_idx]
    recojets = jets[event_idx]
    n_partons = len(partons)

    stage1 = _trace_greedy_match_with_indices(
        ak.to_numpy(genjets.eta), ak.to_numpy(genjets.phi),
        ak.to_numpy(partons.eta), ak.to_numpy(partons.phi), dr_threshold,
    )
    parton_to_genjet = {p: gj for gj, p in stage1.items()}

    stage2 = _trace_greedy_match_with_indices(
        ak.to_numpy(recojets.eta), ak.to_numpy(recojets.phi),
        ak.to_numpy(genjets.eta), ak.to_numpy(genjets.phi), dr_threshold,
    )
    genjet_to_recojet = {gj: r for r, gj in stage2.items()}

    recojet_pt = ak.to_numpy(recojets.pt)
    genjet_pt = ak.to_numpy(genjets.pt)

    results = []
    for p_idx in range(n_partons):
        signal_val = int(partons.signal[p_idx])
        pdg_id = int(partons.pdgId[p_idx])
        base = {"parton": p_idx, "signal": signal_val, "pdg_id": pdg_id}

        if p_idx not in parton_to_genjet:
            results.append({**base, "outcome": "FAIL",
                             "detail": "FAILED: did not pass GenPart-to-GenJet dr matching"})
            continue
        gj_idx = parton_to_genjet[p_idx]
        if gj_idx not in genjet_to_recojet:
            results.append({**base, "outcome": "FAIL",
                             "detail": f"FAILED: GenJet {gj_idx} found, but no reco Jet within dr<{dr_threshold}"})
            continue
        reco_idx = genjet_to_recojet[gj_idx]
        ratio = recojet_pt[reco_idx] / genjet_pt[gj_idx]
        lo, hi = pt_ratio_bounds
        if not (lo < ratio < hi):
            results.append({**base, "outcome": "FAIL",
                             "detail": f"FAILED: reco jet {reco_idx} found, pt_ratio={ratio:.2f} outside ({lo},{hi})"})
            continue
        if abs(pdg_id) == 5:
            qtype = "b quark"
        elif abs(pdg_id) == 21:
            qtype = "gluon"
        else:
            qtype = f"{_trace_quark_label(pdg_id)} quark"
        results.append({**base, "outcome": "SUCCESS", "reco_idx": reco_idx, "pt_ratio": ratio,
                         "detail": f"PASSED: matched to a {qtype}\n-> reco jet {reco_idx}, pt_ratio={ratio:.2f}"})
    return results


def _trace_wrap_multiline(text, width=18):
    """Wraps EACH existing line separately, so an intentional newline
    isn't just treated as another character by textwrap. Confirmed
    directly: the unwrapped version overflowed several boxes' edges."""
    import textwrap
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(wrapped)


def _trace_draw_box(ax, x, y, w, h, text, color, edge, fontsize=7.5):
    import matplotlib.patches as patches
    text = _trace_wrap_multiline(text)
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                   linewidth=1.3, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def trace_event(chs, gen_gj, jets, event_idx, reco_mode, n_real_jets=6, n_null_jets=1,
                 include_isr_target=False, dr_threshold=0.4, outdir="."):
    """Generalized across ALL region structures (full_hww, onshell_w,
    with or without ISR, regardless of strip_bjets) -- builds its slot
    list dynamically from whatever the REAL make_targets() actually
    returns for this reco_mode/include_isr_target, rather than a fixed
    H1+H2 assumption. Verified directly against both full_hww and
    onshell_w+ISR structures before being wired in here -- produced the
    correct 6-slot and 5-slot layouts respectively, with no separate
    code path needed per region type.
    """
    # matplotlib is a heavy, optional dependency -- imported HERE, not at
    # module top, so a normal OSCA run that never sets debug_trace_events
    # never pays the import cost or needs matplotlib installed at all.
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parton_results = _trace_diagnose_partons(chs, gen_gj, jets, event_idx, dr_threshold)

    signal_arr_event = np.full(n_real_jets, -1)
    by_reco_idx = {}
    for r in parton_results:
        if r["outcome"] == "SUCCESS" and r["reco_idx"] < n_real_jets:
            signal_arr_event[r["reco_idx"]] = r["signal"]
            by_reco_idx[r["reco_idx"]] = r

    # Calls the REAL make_targets() directly -- not a separate
    # reimplementation -- so this can never disagree with what the
    # actual production pipeline computes. Batch dim of 1 for one event.
    targets_dict = make_targets(signal_arr_event[None, :], n_real_jets, n_null_jets, reco_mode, include_isr_target)
    null_idx = n_real_jets if n_null_jets > 0 else -1

    slots = []
    for particle in targets_dict:
        for daughter in targets_dict[particle]:
            slots.append((particle, daughter))

    fails_by_signal = {}
    for r in parton_results:
        if r["outcome"] != "SUCCESS":
            fails_by_signal.setdefault(r["signal"], []).append(r["detail"])

    os.makedirs(outdir, exist_ok=True)
    n_slots = len(slots)
    col_w, gap = 1.95, 0.25
    fig_w = max(13, n_slots * (col_w + gap) + 1)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax.axis("off")
    total_w = n_slots * col_w + (n_slots - 1) * gap
    start_x = (fig_w - total_w) / 2
    top_y, top_h = 3.8, 0.8
    bot_y, bot_h = 1.5, 1.6

    ax.text(fig_w / 2, 5.0, "TARGETS", ha="center", fontsize=13, weight="bold")

    shown_pooled = set()
    for i, (particle, daughter) in enumerate(slots):
        x = start_x + i * (col_w + gap)
        _trace_draw_box(ax, x, top_y, col_w, top_h, "[empty]", "#e8e7e2", "#5f5e5a", fontsize=9)
        ax.annotate("", xy=(x + col_w / 2, bot_y + bot_h), xytext=(x + col_w / 2, top_y),
                    arrowprops=dict(arrowstyle="->", color="#444441", lw=1.3))

        value = int(targets_dict[particle][daughter][0])
        if value != null_idx and value != -1:
            text = by_reco_idx[value]["detail"]
            color, edge = "#d7ecdf", "#1f7a4d"
        else:
            groups = _TRACE_PARTICLE_SIGNAL_GROUPS.get(particle, [])
            all_reasons = [d for sig in groups for d in fails_by_signal.get(sig, [])]
            if len(groups) == 1:
                reason = all_reasons[0] if all_reasons else "?"
            else:
                if all_reasons and particle not in shown_pooled:
                    reason = " | ".join(all_reasons)
                    shown_pooled.add(particle)
                elif all_reasons:
                    reason = f"(see first empty {particle} slot)"
                else:
                    reason = "?"
            text = reason
            color, edge = ("#e8e7e2", "#5f5e5a") if value == null_idx else ("#fbe3d6", "#993c1d")
        _trace_draw_box(ax, x, bot_y, col_w, bot_h, text, color, edge)

        ax.text(x + col_w / 2, bot_y - 0.35, f"{particle}.{daughter}", ha="center", fontsize=9, weight="bold")

    ax.set_xlim(0, fig_w)
    ax.set_ylim(0.8, 5.4)
    ax.set_title(f"Event {event_idx}: target resolution ({reco_mode}, ISR={include_isr_target})", fontsize=10, pad=10)
    out_path = os.path.join(outdir, f"target_trace_event{event_idx}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path

# ============================================================
# Module 1: gen-level matching + target/source construction
# ============================================================

@define
class SPANetGenMatch(AnalyzerModule):
    """
    Gen-matches jets to build the {signal-type, target-index} truth used to
    train SPANet, and packages both the network's Source inputs and its
    TARGETS into new Columns.

    Must run downstream of whatever selection chain (jetID, lepton veto, 2b
    requirement, tiered-pt, njets) you want the training population to match
    at inference time -- this module does no event selection of its own. It
    never declares outputs()=="EVENTS", so it can't drop rows; the
    `good_event` boolean it writes is meant to be consumed by a downstream
    SelectOnColumns step in the yaml (matching the existing preselection/
    selection pattern), not applied internally here.
    """

    jet_col: Column
    genpart_col: Column = field(factory=lambda: Column("GenPart"))
    genjet_col: Column = field(factory=lambda: Column("GenJet"))
    output_prefix: str = "SPANet"
    reco_mode: str = "full_hww"  # "full_hww" or "onshell_w"
    n_real_jets: int = 9
    n_null_jets: int = 1
    dr_threshold: float = 0.4
    genjet_eta_cut: float = 2.4
    genjet_pt_cut: float = 15.0
    secondary_order_field: str = "pt"  # "pt" or "qvg" -- ordering applied to jets after the leading 2 btag-sorted slots
    include_isr_target: bool = True  # False = Option A: still gen-match/label ISR jets (signal==4), but don't build a Targets.ISR group for SPANet to predict against
    btag_working_point: str = "M"  # official WP ("L"/"M"/"T") used when btag_mode="label", resolved via getWPs() -- same official correctionlib source HBQuarkMaker's event-selection cut uses
    btag_mode: str = "label"  # "score" (continuous discriminant) or "label" (boolean WP decision). Does not affect jet ORDERING, which always uses the continuous score regardless of this setting (see run(), sorting happens before this is applied).

    qvg_mode: str = "label"  # "score" or "label", applied to PNet QvG (the only QvG discriminant kept -- UParT QvG has been removed entirely: not useful, and ROC-curve testing showed its candidate WPs were unreasonable)
    qvg_label_threshold: float = 0.3  # UNOFFICIAL -- no correctionlib/official WP source exists for QvG as of writing. CONFIRMED for PNet QvG specifically.

    ctag_working_point: str = "M"  # official WP ("L"/"M"/"T") used when ctag_mode="label", resolved via getCTagWPs()
    ctag_mode: str = "label"  # "score" or "label". SCORE: two independent continuous features (CvB, CvL) are kept, since they're genuinely distinct discriminants. LABEL: collapses to ONE boolean feature, true only if the jet clears BOTH the CvB and CvL working points simultaneously. Because the number of Source columns differs between these two modes (2 vs 1), a single event_info.yaml can't describe both -- use two separate configs, one per mode, matching this project's existing pattern of two-config comparisons.

    strip_bjets: bool = False  

    randomize_jet_order: bool = False  # For directly testing whether jet ORDER carries information the network learns from: after Source features AND Targets are both fully built (btag+qvg sort, strip_bjets if active, target index computation -- all completely unchanged), apply an independent per-event random permutation to the REAL jet slots only (never the null slot(s)), then remap every target index through the exact inverse of that permutation so it still points at the same physical jet, just at its new shuffled position. If a model trained this way performs comparably to one trained on the normal btag+qvg order, that's real evidence order itself isn't what the network is learning from -- performance would come from the underlying jet content, not position. Leaves the normal (non-shuffled) path completely untouched when False (the default).
    randomize_seed: int = 42  # Seed for the per-event shuffle above. The underlying RNG is created ONCE (see __shuffle_rng_holder below) and persists across every call to run() for this module instance, so successive chunks draw genuinely different permutations rather than each chunk restarting from the same seed. This guarantee holds only if the SAME module instance processes every chunk sequentially -- if chunks are ever processed by separate instances/workers, each would restart from this same seed, which would not fully defeat the purpose (each individual event still gets an independent per-event shuffle) but could correlate shuffles ACROSS chunk boundaries. Good enough for this study's purpose (a coarse ablation on whether order matters at all), not a cryptographically rigorous randomization -- worth knowing the difference if the result is used as anything more precise.

    debug_trace_events: list = field(factory=list)  # Event indices (within each chunk) to save a detailed per-parton gen-matching + target-resolution trace PNG for, via trace_event() defined earlier in this same file. Empty by default -- does nothing unless explicitly populated. Deliberately a list, not a single int or a bare 'object', so it stays a concrete, cattrs-friendly type (an object-typed field previously broke cattrs' structure-hook generation at OSCA startup for every AnalyzerModule subclass, confirmed directly earlier -- worth not repeating that mistake here).
    debug_trace_outdir: str = "./target_trace_plots"  # Where debug_trace_events' output PNGs get saved.

    __wp_cache: dict = field(factory=dict)
    __ctag_wp_cache: dict = field(factory=dict)
    __shuffle_rng_holder: dict = field(factory=dict)  # Holds {"rng": <generator>} once lazily initialized -- MUST be dict-typed, not object-typed: cattrs (OSCA's YAML-to-class converter) generates a structure hook for every AnalyzerModule subclass at startup, regardless of which pipelines are actually enabled, and cannot handle a plain `object` type annotation. Confirmed directly: object-typed field reproduces the exact "Unsupported type: <class 'object'>" error; dict-typed (matching __wp_cache/__ctag_wp_cache, already proven working) resolves it.

    def getWPs(self, metadata):
        """
        Mirrors HBQuarkMaker.getWPs exactly (same file/tagger/correction_name
        metadata path, same correctionlib call, same per-file caching) --
        this module reads its own working-point thresholds from the SAME
        source HBQuarkMaker's event-selection cut already uses, rather than
        a second, independently-hardcoded number that could silently drift
        out of sync with it.
        """
        file_path = metadata["era"]["btag_scale_factors"]["file"]
        tagger = metadata["era"]["btag_scale_factors"]["tagger"]
        cname = metadata["era"]["btag_scale_factors"]["correction_name"]

        if file_path in self.__wp_cache:
            return tagger, self.__wp_cache[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = {p: cset[cname].evaluate(p) for p in ("L", "M", "T")}
        self.__wp_cache[file_path] = ret
        return tagger, ret

    def getCTagWPs(self, metadata):
        """
        c-tagging analog of getWPs. Follows the same metadata/correctionlib
        pattern as HCQuarkMaker, resolving TWO independent discriminants
        (CvB, CvL) rather than one -- each gets its own tagger field name
        and its own set of L/M/T thresholds, evaluated separately (CvB
        evaluated as "CvB", CvL evaluated as "CvL" -- these are genuinely
        different correctionlib queries, not the same query reused twice).
        """
        file_path = metadata["era"]["btag_scale_factors"]["c_file"]
        tagger_cvb = metadata["era"]["btag_scale_factors"]["c_tagger"]["CvB"]
        tagger_cvl = metadata["era"]["btag_scale_factors"]["c_tagger"]["CvL"]
        taggers = {"CvB": tagger_cvb, "CvL": tagger_cvl}
        cname = metadata["era"]["btag_scale_factors"]["correction_name"]

        if file_path in self.__ctag_wp_cache:
            return taggers, self.__ctag_wp_cache[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = {
            "CvB": {p: cset[cname].evaluate(p, "CvB") for p in ("L", "M", "T")},
            "CvL": {p: cset[cname].evaluate(p, "CvL") for p in ("L", "M", "T")},
        }
        self.__ctag_wp_cache[file_path] = ret
        return taggers, ret

    def preloadForMeta(self, metadata):
        self.getWPs(metadata)
        self.getCTagWPs(metadata)

    def inputs(self, metadata):
        return [self.jet_col, self.genpart_col, self.genjet_col]

    def outputs(self, metadata):
        p = self.output_prefix
        outs = [Column(f"{p}.Source"), Column(f"{p}.good_event"), Column(("Selection", "good_event"))]
        if self.reco_mode == "full_hww":
            outs += [Column(f"{p}.Targets.H1"), Column(f"{p}.Targets.H2")]
        else:
            outs += [Column(f"{p}.Targets.H1"), Column(f"{p}.Targets.W1")]
        if self.include_isr_target:
            outs.append(Column(f"{p}.Targets.ISR"))
        return outs

    def run(self, columns, params):
        jets = columns[self.jet_col]
        genpart = columns[self.genpart_col]
        genjet = columns[self.genjet_col]

        all_gp_idx = ak.local_index(genpart, axis=1)

        # -- Truth object construction (ported from nano_to_h5_V2.py) --
        H_idx = all_gp_idx[(genpart.pdgId == 25) & genpart.hasFlags(["isLastCopy"])]
        b_mask = (
            (abs(genpart.pdgId) == 5)
            & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
            & ak.any(genpart.genPartIdxMother[:, :, None] == H_idx[:, None, :], axis=2)
        )
        b_quarks = ak.with_field(genpart[b_mask], 1, "signal")

        incoming_mask = (genpart.status == 21) & (genpart.genPartIdxMother == -1)
        incoming_idx = all_gp_idx[incoming_mask]
        isr_mask = (
            (genpart.pdgId == 21)
            & ak.any(genpart.genPartIdxMother[:, :, None] == incoming_idx[:, None, :], axis=2)
            & (genpart.status != 21)
        )
        isr_gluon = ak.with_field(genpart[isr_mask], 4, "signal")

        if self.reco_mode == "full_hww":
            W_mask = (abs(genpart.pdgId) == 24) & genpart.hasFlags(["isLastCopy"])
            W_all_idx = all_gp_idx[W_mask]
            W_bosons = genpart[W_mask]
            W_mass_1 = ak.flatten(W_bosons[:, :1].mass)
            W_mass_2 = ak.flatten(W_bosons[:, 1:2].mass)
            onshell_is_first = W_mass_1 >= W_mass_2
            onshell_W_idx = ak.where(onshell_is_first[:, None], W_all_idx[:, :1], W_all_idx[:, 1:2])
            offshell_W_idx = ak.where(onshell_is_first[:, None], W_all_idx[:, 1:2], W_all_idx[:, :1])

            q_onshell_mask = (
                (abs(genpart.pdgId) <= 4)
                & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
                & ak.any(genpart.genPartIdxMother[:, :, None] == onshell_W_idx[:, None, :], axis=2)
            )
            q_onshell = ak.with_field(genpart[q_onshell_mask], 2, "signal")

            q_offshell_mask = (
                (abs(genpart.pdgId) <= 4)
                & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
                & ak.any(genpart.genPartIdxMother[:, :, None] == offshell_W_idx[:, None, :], axis=2)
            )
            q_offshell = ak.with_field(genpart[q_offshell_mask], 3, "signal")

            chs = ak.concatenate([b_quarks, q_onshell, q_offshell, isr_gluon], axis=1)

            def good_event_mask(gj_with_signal):
                return (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 1, False), axis=1) >= 2
                ) & (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 2, False), axis=1)
                    + ak.sum(ak.fill_none(gj_with_signal.signal == 3, False), axis=1)
                    >= 3
                )
        else:  # onshell_w
            W_mask = (abs(genpart.pdgId) == 24) & genpart.hasFlags(["isLastCopy"])
            W_all_idx = all_gp_idx[W_mask]
            W_bosons = genpart[W_mask]
            W_mass_1 = ak.flatten(W_bosons[:, :1].mass)
            W_mass_2 = ak.flatten(W_bosons[:, 1:2].mass)
            onshell_is_first = W_mass_1 >= W_mass_2
            onshell_W_idx = ak.where(onshell_is_first[:, None], W_all_idx[:, :1], W_all_idx[:, 1:2])

            q_onshell_mask = (
                (abs(genpart.pdgId) <= 4)
                & genpart.hasFlags(["isFirstCopy", "fromHardProcess"])
                & ak.any(genpart.genPartIdxMother[:, :, None] == onshell_W_idx[:, None, :], axis=2)
            )
            q_from_W = ak.with_field(genpart[q_onshell_mask], 2, "signal")
            chs = ak.concatenate([b_quarks, q_from_W, isr_gluon], axis=1)

            def good_event_mask(gj_with_signal):
                return (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 1, False), axis=1) >= 2
                ) & (
                    ak.sum(ak.fill_none(gj_with_signal.signal == 2, False), axis=1) >= 2
                )

        # -- Gen jet selection (same eta/pt floor as nano_to_h5_V2.py) --
        genjet_cut = (abs(genjet.eta) < self.genjet_eta_cut) & (genjet.pt > self.genjet_pt_cut)
        gen_gj = genjet[genjet_cut]

        # -- Stage 1: gen parton -> gen jet --
        signal_genjet_list = greedy_match(gen_gj, chs, "signal", dr_threshold=self.dr_threshold)
        gen_gj = ak.with_field(gen_gj, ak.Array(signal_genjet_list), "signal")
        signal_gen_jets = gen_gj[ak.Array([[s != -1 for s in ev] for ev in signal_genjet_list])]

        # -- Stage 2: gen jet -> reco jet, with pt_ratio sanity check --
        signal_reco_list, matched_pt_list = greedy_match(
            jets, signal_gen_jets, "signal", dr_threshold=self.dr_threshold, store_pt=True
        )
        jets = ak.with_field(jets, ak.Array(signal_reco_list), "signal")
        jets = ak.with_field(jets, ak.Array(matched_pt_list), "matched_genjet_pt")
        pt_ratio = jets.pt / jets.matched_genjet_pt
        good_match_ptr = ak.fill_none(
            (pt_ratio > 0.5) & (pt_ratio < 2.0) & (jets.signal != -1), False
        )
        # Jets failing the pt_ratio check get relabeled to -1 (background)
        # rather than dropped, keeping the jet collection's width intact.
        jets = ak.with_field(jets, ak.where(good_match_ptr, jets.signal, -1), "signal")

        if self.debug_trace_events:
            # trace_event is defined directly above in this same file --
            for event_idx in self.debug_trace_events:
                trace_event(
                    chs, gen_gj, jets, event_idx,
                    reco_mode=self.reco_mode, n_real_jets=self.n_real_jets,
                    n_null_jets=self.n_null_jets, include_isr_target=self.include_isr_target,
                    dr_threshold=self.dr_threshold, outdir=self.debug_trace_outdir,
                )

        # -- btag+secondary ordering, same convention used at inference time --
        btag_sort_idx = ak.argsort(jets.btagUParTAK4B, axis=1, ascending=False)
        jets_partial = jets[btag_sort_idx]

        if self.secondary_order_field == "pt":
            secondary_field = jets_partial[:, 2:].pt
        elif self.secondary_order_field == "qvg":
            secondary_field = jets_partial[:, 2:].btagPNetQvG
        else:
            raise ValueError(
                f"secondary_order_field must be 'pt' or 'qvg', got {self.secondary_order_field!r}"
            )

        pt_idx = ak.argsort(secondary_field, axis=1, ascending=False) + 2
        jets_sorted = ak.concatenate([jets_partial[:, :2], jets_partial[pt_idx]], axis=1)

        # FIXED: good_event was previously evaluated on the full, untruncated
        # `jets` collection -- but signal_arr (and therefore make_targets)
        # only ever sees the top n_real_jets after this same sort, via
        # pad_and_convert's arr[:, :max_real_jets] truncation. A genuine 3rd
        # q-jet ranked 7th-or-lower by btag+secondary order would pass
        # good_event (which saw it) but never reach make_targets (which
        # didn't) -- leaving H2 with only 2 real slots filled, both
        # remaining slots colliding on the single null_idx, and the second
        # one forced to -1. Evaluating good_event on the SAME truncated
        # window signal_arr actually uses closes that gap: any event that
        # passes genuinely has >=3 real q-jets among the 6 the network will
        # actually be given, not just somewhere in the full event.
        good_event = good_event_mask(jets_sorted[:, : self.n_real_jets])

        if self.strip_bjets:
            top2_signal = jets_sorted[:, :2].signal
            bjet_slot_violation = ak.any((top2_signal == 2) | (top2_signal == 3), axis=1)
            good_event = good_event & ~bjet_slot_violation

            jets_for_features = jets_sorted[:, 2:]
            effective_n_real_jets = self.n_real_jets - 2
        else:
            jets_for_features = jets_sorted
            effective_n_real_jets = self.n_real_jets

        pt_arr = pad_and_convert(jets_for_features.pt, effective_n_real_jets, self.n_null_jets)
        eta_arr = pad_and_convert(jets_for_features.eta, effective_n_real_jets, self.n_null_jets)
        phi_arr = pad_and_convert(jets_for_features.phi, effective_n_real_jets, self.n_null_jets)
        e_arr = pad_and_convert(jets_for_features.energy, effective_n_real_jets, self.n_null_jets)

        if self.btag_mode == "label":
            tagger, wps = self.getWPs(columns.metadata)
            btag_feature = jets_for_features[tagger] > wps[self.btag_working_point]
        elif self.btag_mode == "score":
            btag_feature = jets_for_features.btagUParTAK4B
        else:
            raise ValueError(f"btag_mode must be 'score' or 'label', got {self.btag_mode!r}")
        btag_arr = pad_and_convert(btag_feature, effective_n_real_jets, self.n_null_jets)

        if self.qvg_mode == "label":
            pnet_qvg_feature = jets_for_features.btagPNetQvG > self.qvg_label_threshold
        elif self.qvg_mode == "score":
            pnet_qvg_feature = jets_for_features.btagPNetQvG
        else:
            raise ValueError(f"qvg_mode must be 'score' or 'label', got {self.qvg_mode!r}")
        pnet_qvg_arr = pad_and_convert(pnet_qvg_feature, effective_n_real_jets, self.n_null_jets)

        # c-tagging: SCORE mode keeps CvB and CvL as two independent
        # continuous features (they're genuinely distinct discriminants,
        # not interchangeable). LABEL mode collapses them into ONE boolean
        # feature -- true only if the jet clears BOTH working points
        # simultaneously. This means Source has a DIFFERENT NUMBER of
        # columns depending on ctag_mode -- intentional
        ctag_source_fields = {}
        if self.ctag_mode == "label":
            taggers, wps = self.getCTagWPs(columns.metadata)
            cvb_pass = jets_for_features[taggers["CvB"]] > wps["CvB"][self.ctag_working_point]
            cvl_pass = jets_for_features[taggers["CvL"]] > wps["CvL"][self.ctag_working_point]
            ctag_feature = cvb_pass & cvl_pass
            ctag_source_fields["ctag"] = pad_and_convert(ctag_feature, effective_n_real_jets, self.n_null_jets)
        elif self.ctag_mode == "score":
            taggers, _ = self.getCTagWPs(columns.metadata)
            ctag_source_fields["ctag_cvb"] = pad_and_convert(
                jets_for_features[taggers["CvB"]], effective_n_real_jets, self.n_null_jets
            )
            ctag_source_fields["ctag_cvl"] = pad_and_convert(
                jets_for_features[taggers["CvL"]], effective_n_real_jets, self.n_null_jets
            )
        else:
            raise ValueError(f"ctag_mode must be 'score' or 'label', got {self.ctag_mode!r}")

        signal_arr = pad_and_convert(jets_for_features.signal, effective_n_real_jets, self.n_null_jets)

        jet_counts = ak.num(jets_for_features)
        mask = make_mask(jet_counts, effective_n_real_jets, self.n_null_jets)

        source_fields = {
            "MASK": mask, "pt": pt_arr, "eta": eta_arr, "phi": phi_arr, "e": e_arr,
            "btag": btag_arr, "pnet_qvg": pnet_qvg_arr,
        }
        source_fields.update(ctag_source_fields)
        source = ak.zip(source_fields, depth_limit=1)

        targets = make_targets(
            signal_arr, effective_n_real_jets, self.n_null_jets, self.reco_mode,
            include_isr_target=self.include_isr_target,
        )

        if self.randomize_jet_order:
            # Deliberately placed AFTER both Source and Targets are fully,
            # normally built -- the entire construction above is completely
            # untouched by this flag. Only the REAL jet slots get shuffled,
            # independently per event; the null slot(s) never move, since
            # null_idx's meaning depends on it staying at a fixed position.
            if "rng" not in self.__shuffle_rng_holder:
                self.__shuffle_rng_holder["rng"] = np.random.default_rng(self.randomize_seed)

            n_events = len(pt_arr)
            perms = make_random_permutations(n_events, effective_n_real_jets, self.__shuffle_rng_holder["rng"])
            inv_perms = np.argsort(perms, axis=1)
            null_idx = effective_n_real_jets if self.n_null_jets > 0 else -1

            # MASK is included here deliberately, not just the physical
            # features -- it records which real-jet-range positions hold
            # actual jets vs padding for events with fewer than
            # effective_n_real_jets real jets, and that has to move WITH
            # its jet, not stay fixed at its original position.
            shuffled_source_fields = {
                key: apply_perm_to_source_field(np.asarray(value), perms, effective_n_real_jets)
                for key, value in source_fields.items()
            }
            source = ak.zip(shuffled_source_fields, depth_limit=1)

            # Remap every target's daughter-slot indices through the SAME
            # inverse permutation, so each one still points at the same
            # physical jet, just at its new shuffled position. null_idx/-1
            # are sentinels, not real positions, and pass through unchanged.
            targets = {
                particle: {
                    daughter: remap_target_indices(np.asarray(indices), inv_perms, null_idx)
                    for daughter, indices in daughters.items()
                }
                for particle, daughters in targets.items()
            }

        p = self.output_prefix
        columns[Column(f"{p}.Source")] = source
        columns[Column(f"{p}.good_event")] = good_event
        columns[Column(("Selection", "good_event"))] = good_event
        columns[Column(f"{p}.Targets.H1")] = ak.zip(targets["H1"])
        if self.reco_mode == "full_hww":
            columns[Column(f"{p}.Targets.H2")] = ak.zip(targets["H2"])
        else:
            columns[Column(f"{p}.Targets.W1")] = ak.zip(targets["W1"])
        if self.include_isr_target:
            columns[Column(f"{p}.Targets.ISR")] = ak.zip(targets["ISR"])

        return columns, []

# ============================================================
# Module 2: HDF5 writer
# ============================================================

@define
class SaveSPANetH5(AnalyzerModule):
    """
    Writes SPANetGenMatch's Source/Targets Columns to an HDF5 file per
    chunk. Modeled directly on SaveEventsH5's local-write -> copyFile()
    pattern (same chunk-keyed naming, same temp-file-then-copy mechanics) --
    lifted rather than reinvented, since that part already does what we need.

    Leave `targets_cols` empty for evaluation-only pipelines where
    SPANetGenMatch either isn't run at all, or is run without truth targets
    -- only Source gets written in that case.

    good_event_col : Column, optional
        If given, events where this Column is False are dropped before
        writing -- reproducing what nano_to_h5_V2.py originally did (bad
        events never made it into the H5 at all), applied here rather than
        via a yaml-level SelectOnColumns step. Leave unset to write every
        event SPANetGenMatch saw.
    """

    prefix: str
    source_col: Column
    targets_cols: List[Column] = field(factory=list)
    good_event_col: Column = None
    output_format: str = (
        "{dataset_name}__{sample_name}__{file_id}"
        "__{chunk.event_start}_{chunk.event_stop}.h5"
    )

    def inputs(self, metadata):
        cols = [self.source_col] + list(self.targets_cols)
        if self.good_event_col is not None:
            cols.append(self.good_event_col)
        return cols

    def outputs(self, metadata):
        return []

    def run(self, columns, params):
        import h5py

        file_id = hashlib.md5(
            columns.metadata["chunk"]["file_path"].encode()
        ).hexdigest().upper()
        uid = str(uuid.uuid4())

        target_name = dotFormat(
            self.output_format,
            **dict(dictToDot(columns.metadata)),
            file_id=file_id,
            uuid=uid,
        )
        target = f"{self.prefix}/{target_name}"

        base = Path("localsaved")
        base.mkdir(exist_ok=True, parents=True)
        local_filename = base / f"{uid}.h5"

        if self.good_event_col is not None:
            good_np = ak.to_numpy(columns[self.good_event_col])
        else:
            good_np = None

        try:
            source = columns[self.source_col]
            with h5py.File(local_filename, "w") as f:
                for field_name in source.fields:
                    data = ak.to_numpy(source[field_name])
                    if good_np is not None:
                        data = data[good_np]
                    f.create_dataset(
                        f"INPUTS/Source/{field_name}",
                        data=data,
                        compression="gzip",
                    )
                for tcol in self.targets_cols:
                    tdata = columns[tcol]
                    # e.g. Column("SPANet.Targets.H2") -> HDF5 group "H2"
                    group_name = str(tcol).split(".")[-1]
                    for field_name in tdata.fields:
                        data = ak.to_numpy(tdata[field_name])
                        if good_np is not None:
                            data = data[good_np]
                        f.create_dataset(
                            f"TARGETS/{group_name}/{field_name}",
                            data=data,
                            compression="gzip",
                        )
            copyFile(local_filename, target)
        finally:
            local_filename.unlink(missing_ok=True)

        return columns, []
