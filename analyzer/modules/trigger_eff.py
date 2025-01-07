import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType
import hist

from .utils.btag_points import getBTagWP
from .utils.logging import getSectorLogger


def makeCutSet(x, s, args):
    return [x[s > a] for a in args]


@MODULE_REPO.register(ModuleType.Producer)
def trigger_eff_objects(columns, params):
    logger = getSectorLogger(params)
    # If all the columns are already present, ie a skim, don't bother running the module anything

    jets = columns.get("Jet")
    fat_jets = columns.get("FatJet")

    fat_jets = fat_jets[(fat_jets.pt > 30) & (abs(fat_jets.eta) < 2.4)]

    el = columns.get("Electron")
    mu = columns.get("Muon")
    good_electrons = el[
        (el.cutBased == 4)
        & (el.pt > 10)
        & (abs(el.eta) < 2.4)
        & (el.miniPFRelIso_all < 0.1)
    ]
    good_muons = mu[
        (mu.mediumId) & (mu.pt > 10) & (abs(mu.eta) < 2.4) & (mu.miniPFRelIso_all < 0.1)
    ]

    good_jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
    near_muon = good_jets.nearest(good_muons, threshold=0.4)
    good_jets = good_jets[ak.is_none(near_muon, axis=1)]

    bwps = getBTagWP(params)
    logger.debug(f"B-tagging workign points are:\n {bwps}")
    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
        [bwps["L"], bwps["M"], bwps["T"]],
    )

    columns.add("good_jets", good_jets, shape_dependent=True)
    columns.add("good_electrons", good_electrons)
    columns.add("good_muons", good_muons)
    columns.add("loose_bs", loose_b)
    columns.add("med_bs", med_b)
    columns.add("tight_bs", tight_b)

    ht = ak.sum(good_jets.pt, axis=1)
    columns.add("HT", ht)


@MODULE_REPO.register(ModuleType.Selection)
def iso_muon(events, params, selector):
    era_info = params.dataset.era
    iso_muon_trigger_name = era_info.trigger_names["IsoMuon"]
    selector.add(f"HLT_IsoMu", events.HLT[iso_muon_trigger_name])


@MODULE_REPO.register(ModuleType.Selection)
def trig_eff_selection(events, params, selector):
    one_muon = ak.num(events.good_muons) == 1
    no_electron = ak.num(events.good_electrons) == 0

    selector.add("one_muon", one_muon)
    selector.add("no_electron", no_electron)


@MODULE_REPO.register(ModuleType.Categorization)
def pass_HT_category(events, params, categories):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    categories.add(
        name="PassHT",
        axis=hist.axis.Integer(0, 2, underflow=False, overflow=False, label="PassHT"),
        values=events.HLT[ht_trigger_name],
    )
