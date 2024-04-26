import awkward as ak
import numpy as np

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi

from .axes import *
from .utils import numMatching


@analyzerModule("signal_hists", depends_on=["objects", "delta_r"], categories="main")
def createSignalHistograms(events, analyzer):
    chi_match_axis = hist.axis.IntCategory(
        [0, 1, 2, 3], name="num_matched_chi", label=r"Number of reco chi jets correct"
    )
    stop_match_axis = hist.axis.IntCategory(
        [0, 1, 2, 3, 4],
        name="num_matched_stop",
        label=r"Number of reco jets matched to stop child",
    )
    num_top_3_are_chi = ak.num(
        events.matched_jet_idx[:, 1:][events.matched_jet_idx[:, 1:] < 3], axis=1
    )
    num_top_4_are_stop = ak.num(
        events.matched_jet_idx[events.matched_jet_idx < 4], axis=1
    )
    num_sub_3_are_chi = ak.num(
        events.matched_jet_idx[:, 1:][
            (events.matched_jet_idx[:, 1:] < 4) & (events.matched_jet_idx[:, 1:] > 0)
        ],
        axis=1,
    )
    analyzer.H(f"num_top_3_jets_matched_chi_children",
        chi_match_axis, num_top_3_are_chi, name="num_top_3_are_chi"
    )
    analyzer.H(f"num_sub_3_jets_matched_chi_children",
        chi_match_axis, num_sub_3_are_chi, name="num_sub_3_are_chi"
    )
    analyzer.H(f"num_top_4_jets_matched_stop_children",
        stop_match_axis, num_top_4_are_stop, name="num_top_4_are_stop"
    )

    idx_axis = hist.axis.IntCategory(
        range(0, 8), name="Jet Idx", label=r"$\chi^{\pm}$ b matched jet idx"
    )
    e = events.matched_jet_idx[:, 1]
    m = ~ak.is_none(e)
    e = e[m]
    analyzer.H(f"chi_b_jet_idx",idx_axis, e, name="chi_b_jet_idx", mask=m)

    idx_axis = hist.axis.IntCategory(
        range(0, 8), name="Jet Idx", label=r"$\tilde{t}$ b matched jet idx"
    )
    e = events.matched_jet_idx[:, 0]
    m = ~ak.is_none(e)
    e = e[m]
    analyzer.H(f"stop_b_jet_idx", idx_axis, e, name="stop_b_jet_idx", mask=m)

    analyzer.H(f"chi_b_dr",
        makeAxis(20, 0, 5, "$\\Delta R$ between $\\chi$ and b from $\\tilde{t}$"),
        events.SignalParticles.chi.delta_r(events.SignalParticles.stop_b),
        name="chi_b_dr",
    )

    analyzer.H(f"chi_b_phi",
        makeAxis(25, 0, 4, "$\\Delta \\phi$ between $\\chi$ and b from $\\tilde{t}$"),
        abs(
            angleToNPiToPi(
                events.SignalParticles.chi.phi - events.SignalParticles.stop_b.phi
            )
        ),
        name="chi_b_phi",
    )

    analyzer.H(f"chi_b_eta",
        makeAxis(25, 0, 4, "$\\Delta \\eta$ between $\\chi$ and b from $\\tilde{t}$"),
        events.SignalParticles.chi.eta - events.SignalParticles.stop_b.eta,
        name="chi_b_phi",
    )

    xchildren = ak.concatenate(
        [
            events.SignalParticles.chi_b[:, np.newaxis],
            events.SignalParticles.chi_d[:, np.newaxis],
            events.SignalParticles.chi_s[:, np.newaxis],
        ],
        axis=1,
    )
    cross_jets = ak.cartesian([xchildren, xchildren], axis=1)
    one, two = ak.unzip(cross_jets)

    dr = one.delta_r(two)
    max_delta_rs = ak.max(dr, axis=1)
    min_delta_rs = ak.min(dr[dr > 0], axis=1)

    deta = abs(one.eta - two.eta)
    max_delta_eta = ak.max(deta, axis=1)
    min_delta_eta = ak.min(deta[deta > 0], axis=1)

    dphi = abs(angleToNPiToPi(one.phi - two.phi))
    max_delta_phi = ak.max(dphi, axis=1)
    min_delta_phi = ak.min(dphi[dphi > 0], axis=1)

    analyzer.H(f"max_chi_child_dr",
        makeAxis(20, 0, 5, "Max $\\Delta R$ between $\\chi$ children"),
        max_delta_rs,
        name="max_chi_child_dr",
    )
    analyzer.H(f"min_chi_child_dr",
        makeAxis(20, 0, 5, "Min $\\Delta R$ between $\\chi$ children"),
        min_delta_rs,
        name="min_chi_child_dr",
    )
    analyzer.H(f"mean_chi_child_dr",
        makeAxis(20, 0, 5, "Mean $\\Delta R$ between $\\chi$ children"),
        ak.mean(dr[dr > 0], axis=1),
        name="mean_chi_child_dr",
    )

    analyzer.H(f"max_chi_child_eta",
        makeAxis(25, 0, 5, "Max $\\Delta \\eta$ between $\\chi$ children"),
        max_delta_eta,
        name="max_chi_child_eta",
    )
    analyzer.H(f"min_chi_child_eta",
        makeAxis(25, 0, 5, "Min $\\Delta \\eta$ between $\\chi$ children"),
        min_delta_eta,
        name="min_chi_child_eta",
    )
    analyzer.H(f"mean_chi_child_eta",
        makeAxis(25, 0, 5, "Mean $\\Delta \\eta$ between $\\chi$ children"),
        ak.mean(deta[deta > 0], axis=1),
        name="mean_chi_child_eta",
    )

    analyzer.H(f"max_chi_child_phi",
        makeAxis(25, 0, 5, "Max $\\Delta \\phi$ between $\\chi$ children"),
        max_delta_phi,
        name="max_chi_child_phi",
    )
    analyzer.H(f"min_chi_child_phi",
        makeAxis(25, 0, 5, "Min $\\Delta \\phi$ between $\\chi$ children"),
        min_delta_phi,
        name="min_chi_child_phi",
    )
    analyzer.H(f"mean_chi_child_phi",
        makeAxis(25, 0, 5, "Mean $\\Delta \\phi$ between $\\chi$ children"),
        ak.mean(dphi[dphi > 0], axis=1),
        name="mean_chi_child_phi",
    )

    analyzer.H(f"stop_pt",
        makeAxis(
            100,
            0,
            1500,
            r"$\tilde{t} p_{T}$",
            unit="GeV",
        ),
        events.SignalParticles.stop.pt,
    )

    all_three_mask = ~ak.any(ak.is_none(events.matched_jet_idx[:, 1:4], axis=1), axis=1)
    analyzer.H(f"max_chi_child_dr_all_three",
        makeAxis(20, 0, 5, "Max $\\Delta R$ between $\\chi$ children"),
        max_delta_rs[all_three_mask],
        name="max_chi_child_dr",
        mask=all_three_mask,
    )
    analyzer.H(f"min_chi_child_dr_all_three",
        makeAxis(20, 0, 5, "Min $\\Delta R$ between $\\chi$ children"),
        min_delta_rs[all_three_mask],
        name="min_chi_child_dr",
        mask=all_three_mask,
    )
    analyzer.H(f"mean_chi_child_dr_all_three",
        makeAxis(20, 0, 5, "Mean $\\Delta R$ between $\\chi$ children"),
        ak.mean(dr[dr > 0], axis=1)[all_three_mask],
        name="mean_chi_child_dr",
        mask=all_three_mask,
    )
    analyzer.H(f"stop_pt_all_three",
        makeAxis(
            100,
            0,
            1500,
            r"$\tilde{t} p_{T}$",
            unit="GeV",
        ),
        events.SignalParticles.stop.pt[all_three_mask],
        mask=all_three_mask,
    )

    return events, analyzer


def makeIdxHist(analyzer, hmaker, idxs, name, axlabel, **kwargs):
    analyer.H(name,
        hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name=name, label=axlabel),
        idxs,
        name=name,
        **kwargs,
    )


@analyzerModule("perfect_hists", categories="main")
def genMatchingMassReco(events, analyzer):
    mask = ~ak.any(ak.is_none(events.matched_jets, axis=1), axis=1)
    all_matched = events.matched_jets[mask]

    analyzer.H(f"mchi_gen_matched",
        makeAxis(
            60,
            0,
            3000,
            r"Mass of Gen Matched $\Delta R < 0.2$ Jets From $\tilde{\chi}^{\pm}$",
            unit="GeV",
        ),
        all_matched[:, 1:4].sum().mass,
        mask=mask,
        name="genmatchedchi",
    )
    analyzer.H(f"mstop_gen_matched",
        makeAxis(
            60,
            0,
            3000,
            r"Mass of Gen Matched $\Delta R < 0.2$ Jets From $\tilde{t}$",
            unit="GeV",
        ),
        all_matched[:, 0:4].sum().mass,
        mask=mask,
        name="Genmatchedm14",
    )
    analyzer.H(f"perfect_matching_check",
        hist.axis.IntCategory(
            [0, 1, 2, 3], name="num_matched_chi", label=r"|GenMatcher $\Delta R < 0.2$|"
        ),
        numMatching(events.matched_jet_idx[:, 1:4], events.matched_jet_idx[:, 1:4]),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(f"perfect_matching_count",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher $\Delta R < 0.2$| ",
        ),
        ak.num(
            events.matched_jet_idx[:, 1:4][
                ~ak.is_none(events.matched_jet_idx[:, 1:4], axis=1)
            ],
            axis=1,
        ),
        name="Number of perfectly matched jets",
    )

    all_three_mask = ~ak.any(ak.is_none(events.matched_jet_idx[:, 1:4], axis=1), axis=1)
    makeIdxHist(
        analyzer,
        hmaker,
        events.matched_jet_idx[:, 1:4][all_three_mask],
        "mchi_gen_matched_idxs",
        "mchi_gen_matched_idxs",
        mask=all_three_mask,
    )

    return events, analyzer
