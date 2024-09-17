import itertools as it

import awkward as ak

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi

from .axes import *
from .utils import numMatching
from analyzer.core import MODULE_REPO, ModuleType


#@analyzerModule("b_hists", categories="main", depends_on=["objects"])
@MODULE_REPO.register(ModuleType.Histogram)
def createBHistograms(events, params, histogram_builder, wp_name="M"):
    mapping = {"M" : "med", "L": "loose", "T" : "tight"}
    bjets = events[f"{mapping}_bs"]

    analyzer.H(f"{wp_name}_bjet_pt", pt_axis, bjets.pt, name="{wp_name} BJet $p_{T}$")
    analyzer.H(f"{wp_name}_nb", b_axis, ak.num(bjets.pt), name="{wp_name} BJet Count")
    for i in range(0, 4):
        mask = ak.num(m_bjets, axis=1) > i
        analyzer.H(
            f"{wp_name}_b_{i}_pt",
            makeAxis(100, 0, 800, f"$p_T$ of rank {i} {wp_name} b jet"),
            m_bjets[mask][:,i].pt,
            mask=mask,
            name=f"{Wp_Name} BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(m_bjets, axis=1) > 1
    top2 = m_bjets[mask]
    b_eta = abs(top2[:, 0].eta - top2[:, 1].eta)
    b_phi = abs(angleToNPiToPi(top2[:, 0].phi - top2[:, 1].phi))
    b_dr = top2[:, 0].delta_r(top2[:, 1])

    analyzer.H(
        f"{wp_name}_bb_eta",
        makeAxis(20, 0, 5, f"$\Delta \eta$ between leading 2 {wp_name} b jets"),
        b_eta,
        mask=mask,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between leading 2 {wp_name} b jets",
    )
    analyzer.H(
        f"{wp_name}_bb_phi",
        makeAxis(25, 0, 4, rf"$\Delta \phi$ between leading {wp_name} b jets"),
        mb_phi,
        mask=mask,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between leading 2 {wp_name} b jets",
    )
    analyzer.H(
        f"{wp_name}_bdr",
        makeAxis(20, 0, 5, rf"$\Delta R$ between leading 2 {wp_name} b jets"),
        mb_dr,
        mask=mask,
        name=rf"{wp_Name} BJet $\Delta R$",
        description=rf"$\Delta R$ between leading 2 {wp_name} $p_T$ b jets",
    )
    inv = top2[:, 0] + top2[:, 1]
    analyzer.H(
        f"{wp_name}_b_m",
        makeAxis(60, 0, 3000, f"$m$ of leading 2 {wp_name} b jets", unit="GeV"),
        inv.mass,
        mask=mask,
    )
    analyzer.H(
        f"{wp_name}_b_pt",
        makeAxis(20, 0, 1000, f"$p_T$ of leading 2 {wp_name} b jets", unit="GeV"),
        inv.pt,
        mask=mask,
    )

    return events, analyzer
