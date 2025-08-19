import copy
from analyzer.core import MODULE_REPO, ModuleType


import awkward as ak
import correctionlib
import pydantic as pyd
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import correctionlib.schemav2 as cs

import logging

logger = logging.getLogger(__name__)

# def addRaw(jets, event_rho, isMC=True):
#     # if isMC:
#     #     jets["pt_gen"] = ak.values_astype(
#     #         ak.fill_none(jets.matched_gen.pt, 0), np.float32
#     #     )
#     return jets


def getCorrKey(base_tag, lvl, algo):
    return "_".join([base_tag, lvl, algo])


# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py


def getJerKeys(jer_name, jet_type):
    return f"{jer_name}_ScaleFactor_{jet_type}", f"{jer_name}_PtResolution_{jet_type}"


class JerConf(pyd.BaseModel):
    smear_path: str
    dr_min: float
    pt_min: float
    gen_jet_name: str
    gen_jet_idx_name: str


def smearJets(jets, jer, jet_type, cset, jer_conf, systematic="nom"):
    resrng_unwrapped = cs.Correction(
        name="resrng",
        description="RandSmear",
        version=1,
        inputs=[
            cs.Variable(name="JetEta", type="float", description="eta"),
            cs.Variable(name="JetPt", type="float", description="pt"),
            cs.Variable(name="EventId", type="int", description="event id"),
        ],
        output=cs.Variable(name="rng", type="real"),
        data=cs.HashPRNG(
            nodetype="hashprng",
            inputs=["JetEta", "JetPt", "EventId"],
            distribution="stdnormal",
        ),
    )
    resrng = correctionlib_wrapper(resrng_unwrapped)

    key_jersmear = "JERSmear"
    sf_jersmear = correctionlib_wrapper(cset_jetsmear[key_jersmear])

    sf_key, res_key = getJerKeys(jer, jet_type)
    sf_corr = correctionlib_wrapper(cset[sf_key])
    sf = sf_corr.evaluate(jets["eta"], systematic)
    res = pt_corr.evaluate(jets["eta"], jets["pt"], jets["rho"])

    gen_jets = events[jer_conf.gen_jet_name]
    gjidx = jer_conf.gen_jet_idx_name
    count = ak.num(gen_jets)
    matched_gen_jets_idx = ak.mask(
        jets[gjidx],
        (jets[gjidx] < count) & (jets[gjidx] != -1),
    )
    matched_objs_mask = ~ak.is_none(matched_gen_jets_idx, axis=1)
    matched_gen_jets = gen_jets[matched_gen_jets_idx]
    matched_jets = ak.mask(jets, matched_objs_mask)

    dpt = ak.abs(matched_gen_jets.pt - matched_jets.pt)

    matched_gen_jets = ak.mask(matched_genjets, dpt < jer_conf.pt_min)
    matched_jets = ak.mask(matched_jets, dpt < jer_conf.pt_min)

    jer_smear_factor = sf_jersmear.evaluate(
        matched_jets.pt, matched_jets.eta, matched_jets.rho, events.event, sf, res
    )
    rand = resrng.evaluate(jets.pt.jets.eta, events.event)
    sqed = sf**2 - 1
    rand_smear = 1 + (rand * res) * ak.sqrt(ak.where(sqed > 0, sqed, 0))
    is_matched = ~ak.is_none(matched_jets.pt, axis=1)
    final_factor = ak.where(is_matched, jer_smear_factor, rand_smear)
    jets_smeared = copy.copy(jets)
    jets_smeared["pt"] = jets["pt"] * final_factor
    jets_smeared["mass"] = jets["mass"] * final_factor
    return jets_smeared


@MODULE_REPO.register(ModuleType.Selection)
def applyJetVetoMap(events):
    pass


def getRho(events, path):
    if isinstance(path, str):
        return events[path]
    else:
        return events[path]


def getKey(base, systematic, jet_type):
    return "_".join([base, systematic, jet_type])


@MODULE_REPO.register(ModuleType.Producer)
def correctedJets(columns, params, jet_type="AK4", systematics=None):
    systematics = systematics or []
    # if not systematics:
    #     raise RuntimeError("Must have at least one systematic")
    jec_params = params.dataset.era.jet_corrections

    corrections_path = jec_params.files[jet_type]
    cset = correctionlib.CorrectionSet.from_file(corrections_path)

    base_key = jec_params.jec[params.dataset.sample_type]
    jet_type = jec_params.jet_names[jet_type]

    jets = columns.Jet
    pt_raw = (1 - jets.rawFactor) * jets.pt
    mass_raw = (1 - jets.rawFactor) * jets.mass
    rho = jets.rho
    systs = {}
    for systematic in systematics:
        k = getKey(base_key, systematic, jet_type)
        logger.info(f"Getting jet correction key {k}")
        corr = cset[k]
        # event_rho = getRho(events, jec_params.rho_name)
        factor = corr.evaluate(jets.eta, pt_raw)
        for shift_name, shift in [("Up", 1), ("Down", -1)]:
            # corrected = ak.copy(jets)

            # fields = {field: jets[field] for field in jets.fields}
            fields["rho"] = rho
            fields["pt"] = pt_raw * factor
            fields["mass"] = mass_raw * factor
            # syst = ak.zip(
            #     fields,
            #     depth_limit=1,
            #     parameters=params,
            #     with_name=flat.layout.parameters["__record__"],
            #     behavior=jets.behavior,
            # )

            systematic_name = f"{systematic}_{shift_name}"
            logger.info(f"Adding jet systematic {systematic_name}")
            systs[systematic_name] = syst

    columns.add("CorrectedJet", jets, systs)


# @MODULE_REPO.register(ModuleType.Producer)
# def testJetCorrection(columns, params):
#     j = columns.Jet
#     columns.add("TestCorrJet", j)  # ), {"up" : j, "down" : j})
#     return
#     fields = j.fields
#     ret = {}
#
#     flat = ak.flatten(j)
#
#     nom = {field: j[field] for field in fields}
#     params = copy.copy(j.layout.parameters)
#     v = j * 3
#     for f in v.fields:
#         nom[f] = v[f]
#
#     nom = ak.zip(
#         nom,
#         depth_limit=1,
#         parameters=params,
#         with_name=flat.layout.parameters["__record__"],
#         behavior=j.behavior,
#     )
#
#     for x in ["Up", "Down"]:
#         out = {field: j[field] for field in fields}
#         params = copy.copy(j.layout.parameters)
#         v = j * 3
#         for f in v.fields:
#             out[f] = v[f]
#         out = ak.zip(
#             out,
#             depth_limit=1,
#             parameters=params,
#             with_name=flat.layout.parameters["__record__"],
#             behavior=j.behavior,
#         )
#         ret[x] = out
#
#     print(nom)
#     print(nom.E)
#     columns.add("TestCorrJet", nom, ret)
