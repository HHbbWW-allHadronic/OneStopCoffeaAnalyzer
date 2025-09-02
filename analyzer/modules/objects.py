import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.btag_points import getBTagWP
from .utils.logging import getSectorLogger


def makeCutSet(x, s, args):
    return [x[s > a] for a in args]


@MODULE_REPO.register(ModuleType.Producer)
def jets_and_ht(columns, params):

    jets = columns.get("Jet")
    gj = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]

    gj = gj[((gj.jetId & 0b100) != 0) & ((gj.jetId & 0b010) != 0)]

    if any(x in params.dataset.era.name for x in ["2016", "2017", "2018"]):
        gj = gj[(gj.pt > 50) | ((gj.puId & 0b10) != 0)]

    good_jets = gj
    ht = ak.sum(good_jets.pt, axis=1)

    columns.add("good_jets", good_jets, shape_dependent=True)
    columns.add("HT", ht)


@MODULE_REPO.register(ModuleType.Producer)
def core_objects(columns, params):
    logger = getSectorLogger(params)
    # If all the columns are already present, ie a skim, don't bother running the module anything

    fat_jets = columns.get("FatJet")

    good_fatjets = fat_jets[(fat_jets.pt > 150) & (abs(fat_jets.eta) < 2.4)]

    bwps = getBTagWP(params)
    logger.info(f"B-tagging workign points are:\n {bwps}")
    good_jets = columns.good_jets

    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
        [bwps["L"], bwps["M"], bwps["T"]],
    )

    el = columns.get("Electron")
    mu = columns.get("Muon")
    good_electrons = el[(el.cutBased == 1) & (el.pt > 10) & (abs(el.eta) < 2.4)]
    good_muons = mu[(mu.looseId) & (mu.pfIsoId == 2) & (abs(mu.eta) < 2.4)]

    columns.add("good_fatjets", good_fatjets)
    columns.add("good_electrons", good_electrons)
    columns.add("good_muons", good_muons)
    columns.add("loose_bs", loose_b)
    columns.add("med_bs", med_b)
    columns.add("tight_bs", tight_b)


@MODULE_REPO.register(ModuleType.Producer)
def semilep_objects(columns, params):
    logger = getSectorLogger(params)
    # If all the columns are already present, ie a skim, don't bother running the module anything

    fat_jets = columns.get("FatJet")

    good_fatjets = fat_jets[(fat_jets.pt > 150) & (abs(fat_jets.eta) < 2.4)]

    bwps = getBTagWP(params)
    logger.info(f"B-tagging workign points are:\n {bwps}")
    good_jets = columns.good_jets

    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
        [bwps["L"], bwps["M"], bwps["T"]],
    )

    el = columns.get("Electron")
    mu = columns.get("Muon")
    good_electrons = el[(el.cutBased >= 3) & (el.pt > 30) & (abs(el.eta) < 2.4)]
    good_muons = mu[
        (mu.pt > 30) & (mu.mediumId) & (mu.pfIsoId == 3) & (abs(mu.eta) < 2.4)
    ]

    columns.add("good_fatjets", good_fatjets)
    columns.add("good_electrons", good_electrons)
    columns.add("good_muons", good_muons)
    columns.add("loose_bs", loose_b)
    columns.add("med_bs", med_b)
    columns.add("tight_bs", tight_b)
