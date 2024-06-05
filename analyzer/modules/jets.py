import itertools as it

import awkward as ak
import dask

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi

from .axes import *
from .objects import b_tag_wps
from .utils import numMatching

@analyzerModule("tiny", categories="main")
def tiny(events, analyzer):
    gj = events.good_jets
    analyzer.H(f"h_njet", nj_axis, ak.num(gj), name="njets")
    return events,analyzer


@analyzerModule("jets", categories="main", depends_on=["objects", "event_level"])
def createJetHistograms(events, analyzer):
    gj = events.good_jets
    analyzer.H(f"h_njet", nj_axis, ak.num(gj), name="njets")
    jet_combos = [(0, 4), (0, 3), (1, 4), (0, 2), (1, 3)]
    co = lambda x: it.combinations(x, 2)

    masses = {}

    for i, j in jet_combos:
        jets = gj[:, i:j].sum()
        masses[(i, j)] = jets.mass
        analyzer.H(f"m{i+1}{j}_pt", 
            makeAxis(
                100,
                0,
                1500,
                f"$p_T ( \\sum_{{n={i+1}}}^{{{j}}} jet_{{n}})$ ",
                unit="GeV",
            ),
            jets.pt,
            name=f"Composite Jet {i+1} to Jet {j} $p_T$",
            description=f"$p_T$ of the sum of jets {i+1} to {j}",
        )
        analyzer.H(f"m{i+1}{j}_eta", 
            makeAxis(20, -5, 5, f"$\eta ( \\sum_{{n={i+1}}}^{{{j}}} ) jet_{{n}}$"),
            jets.eta,
            name=rf"Composite Jet {i+1} to Jet {j} $\eta$",
            description=rf"$\eta$ of the sum of jets {i+1} to {j}",
        )
        mtitle = 4 if j - i == 4 else 3
        analyzer.H(rf"m{i+1}{j}_m", 
            makeAxis(60, 0, 3000, f"$m_{{{mtitle}}}$", unit="GeV"),
            jets.mass,
            name=rf"Composite Jet {i+1} to Jet {j} mass",
            description=rf"Mass of the sum of jets {i+1} to {j}",
        )

    for p1, p2 in co(jet_combos):
        p1_1, p1_2 = p1
        p2_1, p2_2 = p2
        mtitle1 = 4 if p1_2 - p1_1 == 4 else 3
        mtitle2 = 4 if p2_2 - p2_1 == 4 else 3
        analyzer.H(f"m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}", 
            [
                makeAxis(
                    60, 0, 3000, rf"$m_{{{mtitle1}}}$", unit="GeV", append_name="1"
                ),
                makeAxis(
                    60, 0, 3000, rf"$m_{{{mtitle2}}}$", unit="GeV", append_name="2"
                ),
            ],
            [masses[p1], masses[p2]],
            name="Comp mass",
        )

        analyzer.H(f"ratio_m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}", 
            [
                makeAxis(60, 0, 3000, rf"$m_{{{mtitle1}}}$", unit="GeV"),
                makeAxis(
                    50,
                    0,
                    1,
                    rf"$\frac{{m_{{ {mtitle2} }} }}{{ m_{{ {mtitle1} }} }}$",
                ),
            ],
            [masses[p1], masses[p2] / masses[p1]],
            name=f"ratio_m{mtitle1}_vs_m{mtitle2}",
        )

    for i in range(0, 4):
        analyzer.H(rf"pt_{i}", 
            makeAxis(60, 0, 3000, f"$p_{{T, {i}}}$", unit="GeV"),
            gj[:, i].pt,
            name=f"$p_T$ of jet {i+1}",
            description=f"$p_T$ of jet {i+1} ",
        )
        analyzer.H(f"eta_{i}", 
            makeAxis(20, 0, 5, f"$\eta_{{{i}}}$"),
            abs(gj[:, i].eta),
            name=f"$\eta$ of jet {i+1}",
            description=f"$\eta$ of jet {i+1}",
        )
        analyzer.H(f"phi_{i}", 
            makeAxis(50, 0, 4, f"$\phi_{{{i}}}$"),
            abs(gj[:, i].phi),
            name=rf"$\phi$ of jet {i+1}",
            description=rf"$\phi$ of jet {i+1}",
        )

    padded_jets = ak.pad_none(gj, 5, axis=1)
    masks = {}
    for i, j in list(x for x in it.combinations(range(0, 5), 2) if x[0] != x[1]):
        mask = ak.num(gj, axis=1) > max(i, j)
        masked_jets = gj[mask]
        d_eta = masked_jets[:, i].eta - masked_jets[:, j].eta
        d_r = masked_jets[:, i].delta_r(masked_jets[:, j])
        d_phi = masked_jets[:, i].phi - masked_jets[:, j].phi
        masks[(i, j)] = mask
        analyzer.H(rf"d_eta_{i+1}_{j}", 
            eta_axis,
            d_eta,
            mask=mask,
            name=rf"$\Delta \eta$ between jets {i+1} and {j}",
            description=rf"$\Delta \eta$ between jets {i+1} and {j}",
        )
        analyzer.H(f"d_phi_{i+1}_{j}", 
            phi_axis,
            d_phi,
            mask=mask,
            name=rf"$\Delta \phi$ between jets {i+1} and {j}",
            description=rf"$\Delta \phi$ between jets {i+1} and {j}",
        )
        analyzer.H(f"d_r_{i+1}_{j}", 
            dr_axis,
            d_r,
            mask=mask,
            name=rf"$\Delta R$ between jets {i+1} and {j}",
            description=rf"$\Delta R$ between jets {i+1} and {j}",
        )

    for i in range(0, 5):
        mask = ak.num(gj, axis=1) > i
        masked_jets = gj[mask]
        htratio = masked_jets[:, i].pt / events.HT[mask]
        analyzer.H(f"pt_ht_ratio_{i+1}", 
            hist.axis.Regular(50, 0, 5, name="pt_o_ht", label=r"$\frac{p_{T}}{HT}$"),
            htratio,
            mask=mask,
            name=rf"Ratio of jet {i} $p_T$ to event HT",
            description=rf"Ratio of jet {i} $p_T$ to event HT",
        )
    for p1, p2 in co(co(range(0, 4))):
        mask = masks[p1] & masks[p2]
        p1_vals = gj[mask][:, p1[0]].phi - gj[mask][:, p1[1]].phi
        p2_vals = gj[mask][:, p2[0]].phi - gj[mask][:, p2[1]].phi
        analyzer.H("d_phi_{}{}_vs_{}{}".format(*p1, *p2), 
            [
                hist.axis.Regular(
                    50, 0, 5, name="dp1", label=r"$\Delta \phi_{" + f"{p1}" + r"}$"
                ),
                hist.axis.Regular(
                    50, 0, 5, name="dp2", label=r"$\Delta \phi_{" + f"{p2}" + r"}$"
                ),
            ],
            [p1_vals, p2_vals],
            mask=mask,
            name=rf"$\Delta \phi_{p1}$ vs $\Delta \phi_{p2}$",
        )
    return events, analyzer


@analyzerModule("other_region_mass_plots", categories="main")
def otherRegionMassHists(events, analyzer):
    hmaker = analyzer.hmaker
    gj = events.good_jets

    jets = gj[:, 0:4].sum()
    mass = jets.mass

    tbs = events.tight_bs
    sr_313_tight_mask = ak.num(tbs, axis=1) >= 3
    filled_tight = ak.pad_none(tbs, 2, axis=1)
    tight_dr = ak.fill_none(filled_tight[:, 0].delta_r(filled_tight[:, 1]), False)
    tight_dr_mask = tight_dr > 1
    sr_313_mask = sr_313_tight_mask & tight_dr_mask

    analyzer.H(rf"313_m4_m", 
        makeAxis(60, 0, 3000, f"$m_{{4}}$", unit="GeV"),
        jets.mass[sr_313_mask],
        name=rf"M4 in the 313 Region",
        description=rf"M4 in the 313 region",
        mask=sr_313_mask,
    )

    return events, analyzer
