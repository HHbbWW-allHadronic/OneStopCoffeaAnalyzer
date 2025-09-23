import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.btag_points import getBTagWP
from .utils.logging import getSectorLogger


def makeCutSet(x, s, args):
    return [x[s > a] for a in args]


@MODULE_REPO.register(ModuleType.Producer)
def jets(
    columns,
    params,
    jet_in,
    jet_out,
    min_pt=30,
    min_eta=2.4,
    include_puid=True,
    include_jetid=True,
):

    jets = columns.get(jet_in)
    gj = jets[(jets.pt > min_pt) & (abs(jets.eta) < min_eta)]

    if include_jetid:
        gj = gj[((gj.jetId & 0b100) != 0) & ((gj.jetId & 0b010) != 0)]

    if include_puid:
        if any(x in params.dataset.era.name for x in ["2016", "2017", "2018"]):
            gj = gj[(gj.pt > 50) | ((gj.puId & 0b10) != 0)]

    good_jets = gj
    columns.add(jet_out, good_jets, shape_dependent=True)


@MODULE_REPO.register(ModuleType.Producer)
def jetmap_vetoed_jets(columns, params, jet_in, jet_out, veto_type="jetvetomap"):

    import correctionlib

    jets = columns.get(jet_in)
    veto_params = params.dataset.era.jet_veto_map
    fname = veto_params.file
    name = veto_params.name
    cset = correctionlib.CorrectionSet.from_file(fname)
    eval_veto = cset[name]
    j = columns[jet_in]
    j = j[
        (abs(j.eta) < 2.4)
        & (j.pt > 15)
        & ((j.jetId & 0b100) != 0)
        & ((j.chEmEF + j.neEmEF) < 0.9)
    ]
    vetoes = eval_veto.evaluate(veto_type, j.eta, j.phi)
    good_jets = j[(vetoes == 0)]

    columns.add(jet_out, good_jets, shape_dependent=True)


@MODULE_REPO.register(ModuleType.Producer)
def ht(
    columns,
    params,
    jet_in,
):

    jets = columns.get(jet_in)
    ht = ak.sum(jets.pt, axis=1)
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
        (mu.pt > 30) & (mu.mediumId) & (mu.pfIsoId >= 3) & (abs(mu.eta) < 2.4)
    ]

    columns.add("good_fatjets", good_fatjets)
    columns.add("good_electrons", good_electrons)
    columns.add("good_muons", good_muons)
    columns.add("loose_bs", loose_b)
    columns.add("med_bs", med_b)
    columns.add("tight_bs", tight_b)
