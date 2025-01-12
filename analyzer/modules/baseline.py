import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType


@MODULE_REPO.register(ModuleType.Selection)
def signal_hlt(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
    selector.add(
        f"HLT_HT | HLT_AK8", events.HLT[ht_trigger_name] | events.HLT[ak8_trigger_name]
    )

@MODULE_REPO.register(ModuleType.Selection)
def hlt_HT(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    selector.add(f"HLT_HT", events.HLT[ht_trigger_name])

@MODULE_REPO.register(ModuleType.Selection)
def hlt_single_jet(events, params, selector):
    era_info = params.dataset.era
    ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
    selector.add(f"HLT_AK8", events.HLT[ak8_trigger_name])

@MODULE_REPO.register(ModuleType.Selection)
def ht_plateau(events, params, selector):
    era_info = params.dataset.era
    ht_plateau = era_info.trigger_plateaus["HT"]
    ht = events.HT
    selector.add(f"HT_Plateau", ht>ht_plateau)

@MODULE_REPO.register(ModuleType.Selection)
def softdrop_plateau(events, params, selector):
    era_info = params.dataset.era
    ak8_sd = era_info.trigger_plateaus["AK8SingleJetPt"]["msoftdrop"]

    filled_fatjets = ak.pad_none(events.good_fatjets, 1, axis=1)
    passes = ak.fill_none(filled_fatjets[:, 0].msoftdrop > ak8_sd, False)
    selector.add(f"SD_Plateau", passes)



@MODULE_REPO.register(ModuleType.Selection)
def single_muon(events, params, selector):
    era_info = params.dataset.era
    single_muon_trigger_name = era_info.trigger_names["SingleMuon"]
    iso_muon_trigger_name = era_info.trigger_names["IsoMuon"]
    selector.add(
        f"HLT_SingleMu | HLT_IsoMu",
        events.HLT[single_muon_trigger_name] | events.HLT[iso_muon_trigger_name],
    )


@MODULE_REPO.register(ModuleType.Selection)
def general_selection(events, params, selector):
    """Signal selection without b cuts"""
    good_jets = events.good_jets
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    passes_highptjet = ak.fill_none(filled_jets[:, 0].pt > 300, False)
    selector.add("highptjet", passes_highptjet)

    passes_jets = (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)
    selector.add("njets", passes_jets)

    passes_0Lep = (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)
    selector.add("0Lep", passes_0Lep)


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
    selector.add("2bjet", passes_2bjet)
    selector.add("1tightbjet", passes_1tightbjet)
    selector.add("b_dr", passes_b_dr)


@MODULE_REPO.register(ModuleType.Selection)
def partial_signal313_selection(events, params, selector):
    """Signal 313 selection without b cuts"""
    tight_b = events.tight_bs
    filled_tight = ak.pad_none(tight_b, 2, axis=1)
    tight_dr = ak.fill_none(filled_tight[:, 0].delta_r(filled_tight[:, 1]), False)
    passes_3tightbjet = ak.num(tight_b) >= 3
    passes_b_dr = tight_dr > 1
    selector.add("3tightbjet", passes_3tightbjet)
    selector.add("b_dr", passes_b_dr)


@MODULE_REPO.register(ModuleType.Selection)
def partial_cr_selection(events, params, selector):
    """Control region selection.
    Requires 0 loose bs.
    """
    loose_b = events.loose_bs
    selector.add("0looseb", (ak.num(loose_b) == 0))

