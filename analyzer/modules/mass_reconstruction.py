import itertools as it

import awkward as ak
import numpy as np

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi

from .axes import *
from .objects import b_tag_wps
from .utils import numMatching


def makeIdxHist(analyzer, idxs, name, axlabel):
    analyzer.H(
        name,
        hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name=name, label=axlabel),
        idxs,
        name=name,
    )


@analyzerModule(
    "reco_efficiency", categories="main", depends_on=["chargino_hists", "combo_mass"]
)
def recoEfficiency(events, analyzer):
    ret = {}
    gj = events.good_jets
    all_three_mask = ~ak.any(ak.is_none(events.matched_jet_idx[:, 1:4], axis=1), axis=1)

    analyzer.H(
        f"m13_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M13|",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 0:3]),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m13_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M13| All Three Matched",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 0:3])[
            all_three_mask
        ],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    analyzer.H(
        f"m24_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M24|",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 1:4]),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m24_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M24| All Three Matched",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 1:4])[
            all_three_mask
        ],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    analyzer.H(
        f"m3_top_3_no_lead_b_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top3NoB|",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b
        ),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m3_top_3_no_lead_b_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top3NoB| All Three Matched",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b
        )[all_three_mask],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )

    analyzer.H(
        f"m3_top_2_plus_lead_b_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top2PlusB|",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_2_plus_lead_b
        ),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m3_top_2_plus_lead_b_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top2PlusB| All Three Matched",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_2_plus_lead_b
        )[all_three_mask],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    analyzer.H(
        f"m3_dr_switched_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ dr switched|",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.delta_r_switched
        ),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m3_dr_switched_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ dr switched| All Three Matched",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.delta_r_switched
        )[all_three_mask],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    analyzer.H(
        f"m3_top_3_no_lead_b_dr_cut_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ top 3 no lead b dr cut|",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b_dr_cut
        ),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m3_top_3_no_lead_b_dr_cut_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ top 3 no lead b dr cut| All Three Matched",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b_dr_cut
        )[all_three_mask],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    return ret


@analyzerModule("combo_mass", depends_on=["chargino_hists"], categories="main")
def combo_method(events, analyzer):
    ret = {}
    gj = events.good_jets

    idx = ak.local_index(gj, axis=1)
    med_bjet_mask = gj.btagDeepFlavB > b_tag_wps[1]

    lead_b_idx = idx[med_bjet_mask][:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]

    jets_idx = ak.argcombinations(gj, 3, axis=1, replacement=False)
    import numpy as np

    one, two, three = ak.unzip(jets_idx)
    jets_idx = ak.concatenate(
        [one[:, :, np.newaxis], two[:, :, np.newaxis], three[:, :, np.newaxis]], axis=2
    )

    jets = ak.combinations(gj, 3, axis=1, replacement=False)
    one, two, three = ak.unzip(jets)
    jets = ak.concatenate(
        [one[:, :, np.newaxis], two[:, :, np.newaxis], three[:, :, np.newaxis]], axis=2
    )

    cross_jets = ak.cartesian([jets, jets], axis=2)
    one, two = ak.unzip(cross_jets)
    max_delta_rs = ak.max(one.delta_r(two), axis=2)

    dr_mask = max_delta_rs < 4
    lead_b_mask = (
        (jets_idx[:, :, 0] == lead_b_idx)
        | (jets_idx[:, :, 1] == lead_b_idx)
        | (jets_idx[:, :, 2] == lead_b_idx)
    )
    sublead_b_mask = (
        (jets_idx[:, :, 0] == sublead_b_idx)
        | (jets_idx[:, :, 1] == sublead_b_idx)
        | (jets_idx[:, :, 2] == sublead_b_idx)
    )

    summed = jets[:, :, 0] + jets[:, :, 1] + jets[:, :, 2]
    all_masses = summed.mass
    complete_mask = (~lead_b_mask) & dr_mask & sublead_b_mask
    just_b_mask = (~lead_b_mask) & sublead_b_mask
    uncomp_mass = ak.where(
        ak.any(complete_mask, axis=1),
        ak.firsts(all_masses[complete_mask]),
        all_masses[just_b_mask][:, 0],
    )
    uncomp_idx = ak.where(
        ak.any(complete_mask, axis=1),
        ak.firsts(jets_idx[complete_mask]),
        jets_idx[just_b_mask][:, 0],
    )

    analyzer.H(
        f"m3_top_3_no_lead_b_delta_r_cut",
        makeAxis(
            60,
            0,
            3000,
            r"mass of jets 1-3 without leading b, $\Delta R < 3$",
            unit="GeV",
        ),
        uncomp_mass,
        name="mass of jets 1-3 without leading b dr cut",
    )

    makeIdxHist(
        analyzer,
        uncomp_idx,
        "m3_top_3_no_lead_b_delta_r_cut_idx",
        "m3_top_3_no_lead_b_delta_r_cut_idxs",
    )
    events["matching_algos", "top_3_no_lead_b_dr_cut"] = uncomp_idx

    # combos = ak.argcombinations(list(range(6)), 3, axis=0)
    # padmass = ak.pad_none(all_masses, len(combos), axis=1)
    return events, analyzer


@analyzerModule("chargino_hists", categories="main")
def charginoRecoHistograms(events, analyzer):
    ret = {}
    gj = events.good_jets

    idx = ak.local_index(gj, axis=1)
    med_bjet_mask = gj.btagDeepFlavB > b_tag_wps[1]

    t_lead_b_idx = idx[med_bjet_mask]
    lead_b_idx = t_lead_b_idx[:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]
    no_lead_idxs = idx[idx != lead_b_idx]
    no_sublead_idxs = idx[idx != sublead_b_idx]
    no_lead_or_sublead_idxs = idx[(idx != sublead_b_idx) & (idx != lead_b_idx)]
    no_lead_jets = gj[no_lead_idxs]
    no_sublead_jets = gj[no_sublead_idxs]
    first, second = ak.unzip(ak.combinations(no_lead_jets[:, 0:3], 2))
    max_dr_no_lead = ak.max(first.delta_r(second), axis=1)
    first, second = ak.unzip(ak.combinations(no_sublead_jets[:, 0:3], 2))
    max_dr_no_sublead = ak.max(first.delta_r(second), axis=1)
    max_no_lead_over_max_sublead = max_dr_no_lead / max_dr_no_sublead

    jets = gj[:, 0:4].sum()
    m14 = jets.mass
    uncomp_charg_idxs = no_lead_idxs[:, 0:3]
    uncomp_charg = (no_lead_jets[:, 0:3].sum()).mass
    m14_axis = makeAxis(60, 0, 3000, r"$m_{14}$ [GeV]")
    mchi_axis = makeAxis(60, 0, 3000, r"$m_{\chi}$ [GeV]")

    analyzer.H(
        f"m3_top_3_no_lead_b",
        makeAxis(60, 0, 3000, r"mass of jets 1-3 without leading b", unit="gev"),
        uncomp_charg,
        name="mass of jets 1-3 without leading b",
    )
    analyzer.H(
        f"m14_vs_m3_top_3_no_lead_b",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, r"$m_{3 (no b)}$", unit="GeV"),
        ],
        [m14, uncomp_charg],
        name="$m_{14}$ vs Mass of Jets 1-3 Without Leading B",
    )

    # ret[f"m3_top_3_no_b_unless_dR_charg_gt_2"] = hmaker(
    #    mchi_axis,
    #    ak.where(
    #        max_no_lead_over_max_sublead > 2,
    #        (no_lead_jets[:, 0:3].sum()).mass,
    #        (
    #            gj[no_lead_or_sublead_idxs][:, 0:2].sum()
    #            + gj[ak.singletons(lead_b_idx)][:, 0]
    #        ).mass,
    #    ),
    #    name="m3_top3_no_b_unless_dR_charg_gt_2",
    # )

    comp_charg = (no_lead_jets[:, 0:2].sum() + gj[ak.singletons(lead_b_idx)][:, 0]).mass
    comp_charg_idxs = ak.concatenate(
        [no_lead_idxs[:, 0:2], ak.singletons(lead_b_idx)], axis=1
    )

    one, two = ak.unzip(ak.cartesian([no_lead_jets[:, 0:3], no_lead_jets[:, 0:3]]))
    dr = one.delta_r(two)
    max_dr_no_lead_b = ak.max(dr, axis=1)
    one, two = ak.unzip(
        ak.cartesian([gj[no_sublead_idxs[:, 0:3]], gj[no_sublead_idxs[:, 0:3]]], axis=1)
    )
    dr = one.delta_r(two)
    max_dr_no_sublead_b = ak.max(dr, axis=1)
    ratio = (max_dr_no_lead_b / max_dr_no_sublead_b) > 2

    decided = ak.where(ratio, comp_charg, uncomp_charg)
    decided_idxs = ak.where(ratio, comp_charg_idxs, uncomp_charg_idxs)

    events["matching_algos"] = ak.zip(
        dict(
            delta_r_switched=decided_idxs,
            top_2_plus_lead_b=comp_charg_idxs,
            top_3_no_lead_b=uncomp_charg_idxs,
        )
    )

    makeIdxHist(
        analyzer,
        uncomp_charg_idxs,
        "m3_top_3_no_lead_b_idxs",
        "m3_top_3_no_lead_b_idxs",
    )
    makeIdxHist(
        analyzer,
        comp_charg_idxs,
        "m3_top_2_plus_lead_b_idxs",
        "m3_top_2_plus_lead_b_idxs",
    )
    #print(comp_charg_idxs)
    #print(decided_idxs)
    #makeIdxHist(
    #    analyzer, decided_idxs, "m3_dr_switched_idxs", "m3_dr_switched_idxs"
    #)

    analyzer.H(
        f"m3_top_2_plus_lead_b",
        makeAxis(
            60, 0, 3000, r"Mass of leading 2 $p_{T}$ Jets + leading b Jet", unit="GeV"
        ),
        comp_charg,
        name="m3_top_2_plus_lead_b",
    )

    analyzer.H(
        f"m3_dr_switched",
        makeAxis(60, 0, 3000, r"$\Delta R$>2 Switched Mass", unit="GeV"),
        comp_charg,
        name="m3_top_2_plus_lead_b_delta_r_switch",
    )

    analyzer.H(
        f"m14_vs_m3_top_2_plus_lead_b",
        [
            makeAxis(60, 0, 3000, r"$m_{4}$", unit="GeV"),
            makeAxis(
                60,
                0,
                3000,
                r"Mass of leading 2 $p_{T}$ Jets + leading b Jet",
                unit="GeV",
            ),
        ],
        [m14, comp_charg],
        name="Mass of Top 2 $p_T$ Jets Plus Leading b Jet",
    )
    ratio_axis = hist.axis.Regular(
        50,
        0,
        1,
        name=f"ratio",
        label=r"$\frac{m_{\chi}}{m_{4}}$ [GeV]",
    )

    analyzer.H(
        f"ratio_m14_vs_m3_top_2_plus_lead_b",
        [
            makeAxis(
                60,
                0,
                3000,
                r"$m_{4}$",
                unit="GeV",
            ),
            makeAxis(
                50,
                0,
                1,
                r"$\frac{m_{3 (incl b)}}{m_{4}}$",
            ),
        ],
        [m14, comp_charg / m14],
        name="ratio_m14_vs_m3_top_2_plus_lead_b",
    )

    analyzer.H(
        f"ratio_m14_vs_m3_top_3_no_lead_b",
        [
            makeAxis(
                60,
                0,
                3000,
                r"$m_{4}$",
                unit="GeV",
            ),
            makeAxis(
                50,
                0,
                1,
                r"$\frac{m_{3 \mathrm{(no b)}}}{m_{4}}$",
            ),
        ],
        [m14, uncomp_charg / m14],
        name="ratio_m3_top_3_no_lead_b",
    )

    return events, analyzer


@analyzerModule("stop_reco", categories="main")
def stopreco(events, analyzer):
    ret = {}
    jets = events.good_jets[:, 0:4].sum()
    analyzer.H(
        f"m14_vs_pt14",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(50, 0, 500, r"$pt_{14}$", unit="GeV"),
        ],
        [jets.mass, jets.pt],
        name="m14 vs pt 14",
    )

    padded_jets = ak.pad_none(events.good_jets, 6, axis=1)

    top5sum = padded_jets[:, 0:5].sum()
    top6sum = padded_jets[:, 0:6].sum()

    fsrincluded = ak.where((top5sum.pt < jets.pt), top5sum.mass, jets.mass)

    analyzer.H(
        "m14_gt100_m15",
        makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
        fsrincluded,
        name="m14 or maybe m15",
    )

    return events, analyzer
