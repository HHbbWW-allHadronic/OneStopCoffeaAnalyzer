import functools
import logging
import operator as op

import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.logging import getSectorLogger


@MODULE_REPO.register(ModuleType.Selection)
def signal_hlt(events, params, selector):
    ht_trigger_name = params["trigger_names"]["HT"]
    ak8_trigger_name = params["trigger_names"]["AK8SingleJetPt"]
    #selector.add(f"hlt_ht", events.HLT[ht_trigger_name], type="or")
    #selector.add(f"hlt_ak8", events.HLT[ak8_trigger_name], type="or")
    #selector.add(f"HLT_HT | HLT_AK8", events.HLT[ht_trigger_name] | events.HLT[ht_trigger_name])
    selector.add(f"HLT_HT | HLT_AK8", events.HLT[ht_trigger_name] | events.HLT[ak8_trigger_name])


@MODULE_REPO.register(ModuleType.Selection)
def signal_muon(events, params, selector):
    single_muon_trigger_name = params["trigger_names"]["SingleMuon"]
    iso_muon_trigger_name = params["trigger_names"]["IsoMuon"]
    selector.add(f"hlt_single_muon", events.HLT[single_muon_trigger_name], type="or")
    selector.add(f"hlt_iso_muon", events.HLT[iso_muon_trigger_name], type="or")


@MODULE_REPO.register(ModuleType.Selection)
def general_selection(events, params, selector):
    """Signal selection without b cuts"""
    good_jets = events.good_jets
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    passes_highptjet = ak.fill_none(filled_jets[:, 0].pt > 300, False)
    selector.add("highptjet", passes_highptjet, type="and")

    passes_jets = (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)
    selector.add("njets", passes_jets, type="and")

    passes_0Lep = (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)
    selector.add("0Lep", passes_0Lep, type="and")


@MODULE_REPO.register(ModuleType.Selection)
def partial_signal312_selection(events, params, selector):
    """Signal selection without b cuts"""
    med_b = events.med_bs
    tight_b = events.tight_bs
    filled_med = ak.pad_none(med_b, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    passes_2bjet = ak.num(med_b) >= 2
    passes_1tightbjet = ak.num(tight_b) >= 1
    passes_b_dr = med_dr > 1
    selector.add("2bjet", passes_2bjet, type="and")
    selector.add("1tightbjet", passes_1tightbjet, type="and")
    selector.add("b_dr", passes_b_dr, type="and")


@MODULE_REPO.register(ModuleType.Selection)
def partial_cr_selection(events, params, selector):
    """Control region selection.
    Required 0 loose bs.
    """
    loose_b = events.loose_bs
    selector.add("0looseb", (ak.num(loose_b) == 0), type="and")


# @analyzerModule("baseline_hists", categories="main")
# def selectionHists(events, analyzer):
#    gj = events.good_jets
#    nj = ak.num(gj, axis=1)
#    analyzer.H("njets", makeAxis(10, 0, 10, f"NJets"), nj)
#
#    mnj = nj >= 1
#    analyzer.H(
#        "pt0",
#        makeAxis(100, 0, 1500, "$p_{T,0}$", unit="GeV"),
#        gj[mnj][:, 0].pt,
#        mask=mnj,
#    )
#
#    good_muons = events.good_muons
#    good_electrons = events.good_electrons
#
#    analyzer.H(
#        "nlep", makeAxis(6, 0, 6, f"NLep"), ak.num(good_electrons) + ak.num(good_muons)
#    )
#
#    med_b = events.med_bs
#    tight_b = events.tight_bs
#    loose_b = events.loose_bs
#
#    nb = ak.num(med_b, axis=1)
#
#    analyzer.H("n_medb", makeAxis(6, 0, 6, f"N Medium b"), nb)
#    analyzer.H("n_tightb", makeAxis(6, 0, 6, f"N Tight b"), ak.num(tight_b))
#    analyzer.H("n_looseb", makeAxis(6, 0, 6, f"N Loose b"), ak.num(loose_b))
#
#    mnb = nb >= 2
#    twob = med_b[mnb]
#    med_dr = twob[:, 0].delta_r(twob[:, 1])
#    analyzer.H("med_b_dr", makeAxis(20, 0, 5, "$\Delta R$"), med_dr, mask=mnb)
#
#    analyzer.H(
#        "HT",
#        makeAxis(60, 0, 3000, "HT", unit="GeV"),
#        events.HT,
#        name="Event HT",
#        description="Sum of $p_T$ of good AK4 jets.",
#    )
#
#    analyzer.H(
#        f"phi_vs_eta",
#        [makeAxis(50, -5.0, 5.0, f"$\eta$"), makeAxis(50, -5.0, 5.0, f"$\phi$")],
#        [gj.eta, gj.phi],
#        name=f"$\eta$ vs $\phi$ of jet ",
#        description=rf"$\eta$ vs $\phi$ of jet ",
#    )
#
#    return events, analyzer


# @analyzerModule(
#    "baseline_nodr",
#    categories="selection",
#    depends_on=["objects"],
#    processing_info={"used_btag_wps": ["M", "T"]},
# )
# def baselineNoDR(events, analyzer):
#    """Baseline selection for the analysis.
#    Applies the following selection:
#    - Jets[0].pt > 300
#    - 4 <= nJets <= 6
#    - 0 leptons
#    - 2 medium bjets, at least one of which is tight
#    """
#
#    selection = analyzer.selection
#    good_jets = events.good_jets
#    # fat_jets = events.FatJet
#    good_muons = events.good_muons
#    good_electrons = events.good_electrons
#    # loose_b = events.loose_bs
#    med_b = events.med_bs
#    tight_b = events.tight_bs
#    # tight_top = events.tight_tops
#    # selection = PackedSelection()
#    filled_jets = ak.pad_none(good_jets, 4, axis=1)
#    # top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
#
#    filled_med = ak.pad_none(med_b, 2, axis=1)
#    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
#    # selection.add("trigger", (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6))
#    hlt_names = analyzer.profile.hlt
#    if "HLT" in events.fields:
#        hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
#        selection.add("hlt", hlt)
#    selection.add("highptjet", (ak.fill_none(filled_jets[:, 0].pt > 300, False)))
#    selection.add("jets", ((ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)))
#    selection.add("0Lep", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)))
#    selection.add("2bjet", (ak.num(med_b) >= 2))
#    selection.add("1tightbjet", (ak.num(tight_b) >= 1))
#    # selection.add("jet_dr", ((top_two_dr < 4) & (top_two_dr > 2)).to_numpy())
#    return events, analyzer


# @analyzerModule("bbpt_selection", categories="selection", depends_on=["objects"])
# def createBBptSelection(events, analyzer):
#    selection = analyzer.selection
#    med_b = events.med_bs
#    filled_med = ak.pad_none(med_b, 2, axis=1)
#    bbpt = ak.fill_none((filled_med[:, 0] + filled_med[:, 1]).pt, False)
#    selection.add("bbpt", (bbpt > 200).to_numpy())
#    return events, analyzer
