import awkward as ak
from analyzer.core import analyzerModule
import functools
import operator as op


@analyzerModule("baseline_selection", categories="selection", depends_on=["objects"])
def createSelection(events, analyzer):
    """Baseline selection for the analysis.
Applies the following selection:
- Jets[0].pt > 300
- 4 <= nJets <= 6
- 0 leptons
- 2 medium bjets, at least one of which is tight
- delta_R(med_bjets[0],med_bjets[1]) > 1
    """

    selection = analyzer.selection
    good_jets = events.good_jets
    fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    med_b = events.med_bs
    tight_b = events.tight_bs
    # selection = PackedSelection()
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    filled_med = ak.pad_none(med_b, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    # selection.add("trigger", (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6))
    hlt_names = analyzer.profile.hlt
    if "HLT" in events.fields:
        hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
        selection.add('hlt',hlt)
    selection.add("highptjet", (ak.fill_none(filled_jets[:, 0].pt > 300, False)))
    selection.add("jets", ((ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)))
    selection.add("0Lep", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)))
    selection.add("2bjet", (ak.num(med_b) >= 2))
    selection.add("1tightbjet", (ak.num(tight_b) >= 1))
    # selection.add("jet_dr", ((top_two_dr < 4) & (top_two_dr > 2)).to_numpy())
    selection.add("b_dr", (med_dr > 1))
    return events, analyzer


@analyzerModule("bbpt_selection", categories="selection", depends_on=["objects"])
def createBBptSelection(events, analyzer):
    selection = analyzer.selection
    med_b = events.med_bs
    filled_med = ak.pad_none(med_b, 2, axis=1)
    bbpt = ak.fill_none((filled_med[:, 0] + filled_med[:, 1]).pt, False)
    selection.add("bbpt", (bbpt > 200).to_numpy())
    return events, analyzer


@analyzerModule("cr_selection", categories="selection", depends_on=["objects"])
def createCRSelection(events, analyzer):
    selection = analyzer.selection
    good_jets = events.good_jets
    fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    med_b = events.med_bs
    tight_b = events.tight_bs
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
    hlt_names = analyzer.profile.hlt
    if "HLT" in events.fields:
        hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
        selection.add('hlt',hlt)
    selection.add("highptjet", (ak.fill_none(filled_jets[:, 0].pt > 300, False)))
    selection.add("jets", ((ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)))
    selection.add("0Lep", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)))
    selection.add("0looseb", (ak.num(loose_b) == 0))
    return events, analyzer
