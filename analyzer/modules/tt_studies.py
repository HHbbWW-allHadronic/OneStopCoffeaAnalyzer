import awkward as ak
import numpy as np
from analyzer.core import MODULE_REPO, ModuleType
import hist


@MODULE_REPO.register(ModuleType.Selection)
def one_lep_hlt(events, params, selector):
    tn = params.dataset.era.trigger_names
    pass_el = events.HLT[tn["EleWPTight"]]
    if tn["EleCaloIdVT"] is not None:
        pass_el = pass_el | events.HLT[tn["EleCaloIdVT"]]
    pass_mu = events.HLT[tn["SingleMuon"]] | events.HLT[tn["IsoMuon"]]
    selector.add(f"Single Lep", pass_mu | (pass_el & ~pass_mu))


@MODULE_REPO.register(ModuleType.Selection)
def single_lep_selection(events, params, selector):
    """Signal selection without b cuts"""
    good_jets = events.good_jets
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    passes_1Lep = (ak.num(good_electrons) + ak.num(good_muons)) == 1
    selector.add("1Lep", passes_1Lep)

    passes_jets = (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)
    selector.add("njets", passes_jets)


@MODULE_REPO.register(ModuleType.Selection)
def high_pt_jet_selection(events, params, selector):
    """Signal selection without b cuts"""
    good_jets = events.good_jets
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    passes_highptjet = ak.fill_none(filled_jets[:, 0].pt > 300, False)
    selector.add("highptjet", passes_highptjet)


@MODULE_REPO.register(ModuleType.Categorization)
def njets_category(columns, params, categories):
    categories.add(
        name="NJets",
        axis=hist.axis.Integer(3, 7, underflow=False, overflow=False, name="NJets"),
        values=ak.num(columns.good_jets),
    )


def topPtReweightData(pt):
    return np.exp(0.0615 - 0.0005 * pt)


def getWeightRatio(new_weights, before_weights):
    if ak.backend(new_weights) == "typetracer":
        touched = ak.Array(
            ak.typetracer.length_zero_if_typetracer(new_weights).layout.to_typetracer(
                forget_length=True
            )
        )
        return touched
    else:
        before_total = ak.sum(before_weights, axis=0)
        after_total = ak.sum(before_weights * new_weights, axis=0)
        ratio = before_total / after_total
        return ratio


@MODULE_REPO.register(ModuleType.Weight)
def top_pt_reweighting(events, params, weight_manager):

    gp = events.GenPart
    tops = gp[(abs(gp.pdgId) == 6) & gp.hasFlags("isLastCopy")]

    weight = np.sqrt(ak.prod(topPtReweightData(tops.pt), axis=1))
    before_weights = weight_manager.weight()
    before_total = ak.sum(before_weights, axis=0)
    after_total = ak.sum(before_weights * weight, axis=0)
    ratio = before_total / after_total
    weight = weight * ratio
    # MUST be weight * ratio rather than ratio * weight to avoid typing issues :|
    weight_manager.add("top_pt_reweighting", weight)
