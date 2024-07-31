import itertools as it

import awkward as ak

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi

from .axes import *
from .utils import numMatching


@analyzerModule("b_hists", categories="main", depends_on=["objects"])
def createBHistograms(events, analyzer):
    l_bjets = events.loose_bs
    analyzer.H(f"loose_bjet_pt", pt_axis, l_bjets.pt, name="Loose BJet $p_{T}$")
    analyzer.H(f"loose_nb", b_axis, ak.num(l_bjets.pt), name="Loose BJet Count")
    for i in range(0, 4):
        mask = ak.num(l_bjets, axis=1) > i
        analyzer.H(
            f"loose_b_{i}_pt",
            makeAxis(50, 0, 800, f"$p_{{T}}$ of rank  {i} loose b jet"),
            l_bjets[mask][:, i].pt,
            mask=mask,
            name=f"Loose BJet {i} $p_{{T}}$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(l_bjets, axis=1) > 1
    top2 = l_bjets[mask]
    lb_eta = abs(top2[:, 0].eta - top2[:, 1].eta)
    lb_phi = abs(angleToNPiToPi(top2[:, 0].phi - top2[:, 1].phi))
    lb_dr = top2[:, 0].delta_r(top2[:, 1])

    analyzer.H(
        f"loose_bb_eta",
        makeAxis(20, 0, 5, "$\\Delta \\eta$ between leading loose b jets"),
        lb_eta,
        mask=mask,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between the two highest rank loose b jets",
    )
    analyzer.H(
        f"loose_bb_phi",
        makeAxis(25, 0, 4, "$\\Delta \\phi$ between leading loose b jets"),
        lb_phi,
        mask=mask,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between the two highest rank loose b jets",
    )
    analyzer.H(
        f"loose_bdr",
        makeAxis(20, 0, 5, "$\\Delta R$ between leading 2 loose b jets"),
        lb_dr,
        mask=mask,
        name=rf"Loose BJet $\Delta R$",
        description=rf"$\Delta R$ between the top 2 $p_T$ b jets",
    )

    m_bjets = events.med_bs
    analyzer.H(f"medium_bjet_pt", pt_axis, m_bjets.pt, name="Medium BJet $p_{T}$")
    analyzer.H(f"medium_nb", b_axis, ak.num(m_bjets.pt), name="Medium BJet Count")
    for i in range(0, 4):
        mask = ak.num(m_bjets, axis=1) > i
        analyzer.H(
            f"medium_b_{i}_pt",
            makeAxis(100, 0, 800, f"$p_T$ of rank {i} medium b jet"),
            m_bjets[mask][:,i].pt,
            mask=mask,
            name=f"Medium BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(m_bjets, axis=1) > 1
    top2 = m_bjets[mask]
    mb_eta = abs(top2[:, 0].eta - top2[:, 1].eta)
    mb_phi = abs(angleToNPiToPi(top2[:, 0].phi - top2[:, 1].phi))
    mb_dr = top2[:, 0].delta_r(top2[:, 1])

    analyzer.H(
        f"medium_bb_eta",
        makeAxis(20, 0, 5, "$\Delta \eta$ between leading 2 medium b jets"),
        mb_eta,
        mask=mask,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between leading 2 medium b jets",
    )
    analyzer.H(
        f"medium_bb_phi",
        makeAxis(25, 0, 4, "$\Delta \phi$ between leading medium b jets"),
        mb_phi,
        mask=mask,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between leading 2 medium b jets",
    )
    analyzer.H(
        f"medium_bdr",
        makeAxis(20, 0, 5, "$\Delta R$ between leading 2 medium b jets"),
        mb_dr,
        mask=mask,
        name=rf"Medium BJet $\Delta R$",
        description=rf"$\Delta R$ between leading 2 medium $p_T$ b jets",
    )
    inv = top2[:, 0] + top2[:, 1]
    analyzer.H(
        f"medium_b_m",
        makeAxis(60, 0, 3000, f"$m$ of leading 2 medium b jets", unit="GeV"),
        inv.mass,
        mask=mask,
        name=rf"medbmass",
    )
    analyzer.H(
        f"medium_b_pt",
        makeAxis(20, 0, 1000, f"$p_T$ of leading 2 medium b jets", unit="GeV"),
        inv.pt,
        name=rf"medbpt",
        mask=mask,
    )

    return events, analyzer


@analyzerModule("b_ordinality_hists", depends_on=["objects"])
def createBHistograms(events, analyzer):
    gj = events.good_jets
    idx = ak.local_index(gj, axis=1)
    bwp = analyzer.profiles.btag_working_points
    med_bjet_mask = gj.btagDeepFlavB > bwps["medium"]
    t_lead_b_idx = idx[med_bjet_mask]
    lead_b_idx = t_lead_b_idx[:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]

    ordax = hist.axis.Regular(10, 0, 10, name="Jet Rank", label=r"Jet Rank")

    analyzer.H(
        f"lead_medium_bjet_ordinality",
        makeAxis(6, 1, 7, "Leading Medium B Jet Rank"),
        lead_b_idx + 1,
        name="Leading $p_{T}$ Medium B Jet Rank",
    )
    analyzer.H(
        f"sublead_medium_bjet_ordinality",
        makeAxis(6, 1, 7, "SubLeading Medium B Jet Rank"),
        sublead_b_idx + 1,
        name="Subleading $p_{T}$ Medium B Jet Rank",
    )

    return events, analyzer
