import awkward as ak
from analyzer.core import analyzerModule
from .btag_points import getBTagWP


def makeCutSet(x, s, args):
    return [x[s > a] for a in args]



@analyzerModule("extra_objects", categories="post_selection")
def extraObjects(events, analyzer):
    return events, analyzer


@analyzerModule("objects", categories="pre_selection")
def createObjects(events, analyzer):
    good_jets = events.Jet[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)]
    fat_jets = events.FatJet[(events.FatJet.pt > 30) & (abs(events.FatJet.eta) < 2.4)]
    bwps = getBTagWP(analyzer.profile)

    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
       [bwps["L"], bwps["M"], bwps["T"]],
    )

    el = events.Electron
    mu = events.Muon


    good_electrons = el[(el.cutBased == 1) & (el.pt > 10) & (abs(el.eta) < 2.4)]
    good_muons = mu[(mu.looseId) & (mu.pfIsoId == 2) & (abs(mu.eta) < 2.4)]

    events["good_jets"] = good_jets
    events["good_electrons"] = good_electrons
    events["good_muons"] = good_muons
    events["loose_bs"] = loose_b
    events["med_bs"] = med_b
    events["tight_bs"] = tight_b

    ht = ak.sum(good_jets.pt, axis=1)
    events["HT"] = ht

    return events, analyzer
