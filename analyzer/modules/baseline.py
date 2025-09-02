import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType
import hist


@MODULE_REPO.register(ModuleType.Selection)
def signal_hlt(events, params, selector):
    era_info = params.dataset.era

    if "signal" in params.dataset.name and "official" in params.dataset.name:
        ht_cutoffs = {
            "2016_preVFP": 900,
            "2016_postVFP": 900,
            "2017": 1050,
            "2018": 1050,
            "2022_preEE": 1050,
            "2022_postEE": 1050,
            "2023_preBPix": 1050,
            "2023_postBPix": 1050,
        }
        ak8_cutoffs = {
            "2016_preVFP": (360, 30),
            "2016_postVFP": (360, 30),
            "2017": (500, None),
            "2018": (400, 30),
            "2022_preEE": (420, 30),
            "2022_postEE": (420, 30),
            "2023_preBPix": (420, 30),
            "2023_postBPix": (420, 30),
        }
        era_name = era_info.name

        ht_cutoff = ht_cutoffs[era_name]
        ak8pt, ak8sd = ak8_cutoffs[era_name]

        jets = events.Jet
        gj = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]

        gj = gj[((gj.jetId & 0b100) != 0) & ((gj.jetId & 0b010) != 0)]
        if any(x in params.dataset.era.name for x in ["2016", "2017", "2018"]):
            gj = gj[(gj.pt > 50) | ((gj.puId & 0b10) != 0)]

        good_jets = gj
        ht = ak.sum(good_jets.pt, axis=1)

        fat_jets = events.FatJet
        good_fatjets = fat_jets[(fat_jets.pt > 150) & (abs(fat_jets.eta) < 2.4)]
        padded_fatjets = ak.pad_none(good_fatjets, 1, axis=1)

        pass_ht = ht > ht_cutoff
        pass_ak8 = padded_fatjets[:, 0].pt > ak8pt
        if ak8sd is not None:
            pass_ak8 = pass_ak8 & (padded_fatjets[:, 0].msoftdrop > ak8sd)

        selector.add(f"HLT_HT | HLT_AK8", pass_ht | pass_ak8)
    else:
        ht_trigger_name = era_info.trigger_names["HT"]
        ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
        selector.add(
            f"HLT_HT | HLT_AK8",
            events.HLT[ht_trigger_name] | events.HLT[ak8_trigger_name],
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
    selector.add(f"HT_Plateau", ht > ht_plateau)


@MODULE_REPO.register(ModuleType.Selection)
def softdrop_plateau(events, params, selector):
    era_info = params.dataset.era
    ak8_sd = era_info.trigger_plateaus["AK8SingleJetPt"]["msoftdrop"]

    filled_fatjets = ak.pad_none(events.good_fatjets, 1, axis=1)
    passes = ak.fill_none(filled_fatjets[:, 0].msoftdrop > ak8_sd, False)
    selector.add(f"SD_Plateau", passes)


@MODULE_REPO.register(ModuleType.Categorization)
def offline_singlejet_category(events, params, categories):
    era_info = params.dataset.era
    ak8_sd = era_info.trigger_plateaus["AK8SingleJetPt"]["msoftdrop"]
    ak8_pt = era_info.trigger_plateaus["AK8SingleJetPt"]["pt"]

    filled_fatjets = ak.pad_none(events.good_fatjets, 1, axis=1)

    first_fjet = filled_fatjets[:, 0]
    passes = ak.fill_none(
        (first_fjet.msoftdrop > ak8_sd) & (first_fjet.pt > ak8_pt), False
    )
    categories.add(
        name=f"PassOfflineSingleJet",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="PassOfflineSingleJet"
        ),
        values=passes,
    )


@MODULE_REPO.register(ModuleType.Categorization)
def offline_ht_category(events, params, categories):
    era_info = params.dataset.era
    ht_plateau = era_info.trigger_plateaus["HT"]
    ht = events.HT
    categories.add(
        name=f"PassOfflineHT",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="PassOfflineHT"
        ),
        values=ht > ht_plateau,
    )


@MODULE_REPO.register(ModuleType.Categorization)
def hlt_ht_trigger_category(events, params, categories):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    categories.add(
        name=f"PassHLTHT",
        axis=hist.axis.Integer(0, 2, underflow=False, overflow=False, name="PassHLTHT"),
        values=events.HLT[ht_trigger_name],
    )


@MODULE_REPO.register(ModuleType.Categorization)
def hlt_singlejet_trigger_category(events, params, categories):
    era_info = params.dataset.era
    ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
    categories.add(
        name=f"PassHLTSingleJet",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="PassHLTSingleJet"
        ),
        values=events.HLT[ak8_trigger_name],
    )


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
def partial_one_b_selection(events, params, selector):
    """Signal selection without b cuts"""
    med_b = events.med_bs
    passes_1bjet = ak.num(med_b) >= 1
    selector.add("1bjet", passes_1bjet)


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
