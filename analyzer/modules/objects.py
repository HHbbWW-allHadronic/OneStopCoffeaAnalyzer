
import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.btag_points import getBTagWP
from .utils.logging import getSectorLogger


def makeCutSet(x, s, args):
    return [x[s > a] for a in args]


@MODULE_REPO.register(ModuleType.Producer)
def core_objects(columns, params):
    logger = getSectorLogger(params)
    logger.debug(f"B-tagging workign points are:\n {bwps}")
    # If all the columns are already present, ie a skim, don't bother running the module anything
    if all(x in events.fields for x in PRODUCED_COLUMNS):
        return


    jets = columns["jets"]
    fat_jets = columns["fat_jets"]

    good_jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
    fat_jets = fat_jets[(fat_jets.pt > 30) & (abs(fat_jets.eta) < 2.4)]

    bwps = getBTagWP(params)
    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
        [bwps["L"], bwps["M"], bwps["T"]],
    )


    el = columns.electrons()
    mu = events.Muon
    good_electrons = el[(el.cutBased == 1) & (el.pt > 10) & (abs(el.eta) < 2.4)]
    good_muons = mu[(mu.looseId) & (mu.pfIsoId == 2) & (abs(mu.eta) < 2.4)]

    columns["good_electrons"] =   good_electrons
    columns["good_muons"] =  good_muons
    columns["loose_bs"] =  loose_b
    columns["med_bs"] =  med_b
    columns["tight_bs"] =  tight_b
    ht = ak.sum(good_jets.pt, axis=1)
    columns["HT"] = ht






