import copy
from rich import print
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


def availableKeys(cset):
    return set(cset.keys()) | set(cset.compound.keys())


# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py


def getJerKeys(jer_name, jet_type):
    return f"{jer_name}_ScaleFactor_{jet_type}", f"{jer_name}_PtResolution_{jet_type}"


class JerConf(pyd.BaseModel):
    smear_path: str
    dr_min: float
    pt_min: float
    gen_jet_name: str
    gen_jet_idx_name: str


# def smearJets(jets, jer, jet_type, cset, jer_conf, systematic="nom"):
#     resrng_unwrapped = cs.Correction(
#         name="resrng",
#         description="RandSmear",
#         version=1,
#         inputs=[
#             cs.Variable(name="JetEta", type="float", description="eta"),
#             cs.Variable(name="JetPt", type="float", description="pt"),
#             cs.Variable(name="EventId", type="int", description="event id"),
#         ],
#         output=cs.Variable(name="rng", type="real"),
#         data=cs.HashPRNG(
#             nodetype="hashprng",
#             inputs=["JetEta", "JetPt", "EventId"],
#             distribution="stdnormal",
#         ),
#     )
#     resrng = correctionlib_wrapper(resrng_unwrapped)
#
#     key_jersmear = "JERSmear"
#     sf_jersmear = correctionlib_wrapper(cset_jetsmear[key_jersmear])
#
#     sf_key, res_key = getJerKeys(jer, jet_type)
#     sf_corr = correctionlib_wrapper(cset[sf_key])
#     sf = sf_corr.evaluate(jets["eta"], systematic)
#     res = pt_corr.evaluate(jets["eta"], jets["pt"], jets["rho"])
#
#     gen_jets = events[jer_conf.gen_jet_name]
#     gjidx = jer_conf.gen_jet_idx_name
#     count = ak.num(gen_jets)
#     matched_gen_jets_idx = ak.mask(
#         jets[gjidx],
#         (jets[gjidx] < count) & (jets[gjidx] != -1),
#     )
#     matched_objs_mask = ~ak.is_none(matched_gen_jets_idx, axis=1)
#     matched_gen_jets = gen_jets[matched_gen_jets_idx]
#     matched_jets = ak.mask(jets, matched_objs_mask)
#
#     dpt = ak.abs(matched_gen_jets.pt - matched_jets.pt)
#
#     matched_gen_jets = ak.mask(matched_genjets, dpt < jer_conf.pt_min)
#     matched_jets = ak.mask(matched_jets, dpt < jer_conf.pt_min)
#
#     jer_smear_factor = sf_jersmear.evaluate(
#         matched_jets.pt, matched_jets.eta, matched_jets.rho, events.event, sf, res
#     )
#     rand = resrng.evaluate(jets.pt.jets.eta, events.event)
#     sqed = sf**2 - 1
#     rand_smear = 1 + (rand * res) * ak.sqrt(ak.where(sqed > 0, sqed, 0))
#     is_matched = ~ak.is_none(matched_jets.pt, axis=1)
#     final_factor = ak.where(is_matched, jer_smear_factor, rand_smear)
#     jets_smeared = copy.copy(jets)
#     jets_smeared["pt"] = jets["pt"] * final_factor
#     jets_smeared["mass"] = jets["mass"] * final_factor
#     return jets_smeared


@MODULE_REPO.register(ModuleType.Selection)
def applyJetVetoMap(events):
    pass


def getRho(events, path):
    if isinstance(path, str):
        return events[path]
    else:
        return events[path]


def getKeyJec(name, jet_type, params):
    jec_params = params.dataset.era.jet_corrections
    jet_type = jec_params.jet_names[jet_type]
    data_mc = "MC" if params.dataset.sample_type == "MC" else "DATA"
    campaign = jec_params.jec.campaign
    version = jec_params.jec.version
    return f"{campaign}_{version}_{data_mc}_{name}_{jet_type}"


def getKeyJer(name, jet_type, params):
    jec_params = params.dataset.era.jet_corrections
    jet_type = jec_params.jet_names[jet_type]
    data_mc = "MC" if params.dataset.sample_type == "MC" else "DATA"
    campaign = jec_params.jer.campaign
    version = jec_params.jer.version
    return f"{campaign}_{version}_{data_mc}_{name}_{jet_type}"


def smearJets(jets, rho, params, cset, jet_type, include_systematics=False):
    jec_params = params.dataset.era.jet_corrections
    systematics = jec_params.jer.systematics
    print(getKeyJer("PtResolution", jet_type, params))
    evaljer = cset[getKeyJer("PtResolution", jet_type, params)]
    evalsf = cset[getKeyJer("ScaleFactor", jet_type, params)]
    cset_jersmear = correctionlib.CorrectionSet.from_file(jec_params.files["smear"])
    evalsmear = cset_jersmear["JERSmear"]

    genjet_idx_col = jec_params.jer.genjet_idx_col

    inputs = {
        "JetEta": jets.eta,
        "JetPt": jets.pt,
        "Rho": rho,
    }
    jer = evaljer.evaluate(*[inputs[inp.name] for inp in evaljer.inputs])

    def smearWithSystematic(jets, systematic_name):
        sf_inputs = inputs | {"systematic": systematic_name}
        sf = evalsf.evaluate(*[sf_inputs[inp.name] for inp in evalsf.inputs])
        genjet_idx = jets[genjet_idx_col]
        valid_genjet_idxs = ak.mask(genjet_idx, genjet_idx >= 0)

        max_genjet_idx = ak.max(valid_genjet_idxs)
        padded_genjets = ak.pad_none(
            jets,
            0 if max_genjet_idx is None else (max_genjet_idx + 1),
        )

        matched_genjet = padded_genjets[valid_genjet_idxs]
        pt_relative_diff = 1 - matched_genjet.pt / jets.pt
        is_matched_pt = abs(pt_relative_diff) < (3 * jer)
        is_matched_pt = ak.fill_none(is_matched_pt, False)

        sf_scaling = 1.0 + (sf - 1.0) * pt_relative_diff

        smear_inputs = inputs | {
            "JER": jer,
            "JERSF": sf,
            "GenPt": ak.where(is_matched_pt, matched_genjet.pt, -1),
            "EventID": ak.values_astype(rho * 10**6, "int64"),
        }

        final_smear = evalsmear.evaluate(
            *[smear_inputs[inp.name] for inp in evalsmear.inputs]
        )
        print(final_smear)

        smeared_jets = ak.with_field(jets, jets.pt * final_smear, "pt")
        smeared_jets = ak.with_field(smeared_jets, jets.mass * final_smear, "mass")
        return smeared_jets

    if include_systematics:
        return smearWithSystematic(jets, "nom"), {
            k + "_JER": smearWithSystematic(jets, k) for k in systematics
        }
    else:
        return smearWithSystematic(jets, "nom")


@MODULE_REPO.register(ModuleType.Producer)
def corrected_jets(
    columns,
    params,
    input_col: str = None,
    output_col: str = None,
    jet_type="AK4",
    do_smearing=False,
):
    jec_params = params.dataset.era.jet_corrections
    systematics = jec_params.jec.systematics

    jets = columns[input_col]

    corrections_path = jec_params.files[jet_type]
    cset = correctionlib.CorrectionSet.from_file(corrections_path)

    pt_raw = (1 - jets.rawFactor) * jets.pt
    mass_raw = (1 - jets.rawFactor) * jets.mass
    rho = (
        columns.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in columns.fields
        else events.Rho.fixedGridRhoFastjetAll
    )

    systs = {}
    for systematic in systematics:
        k = getKeyJec(systematic, jet_type, params)
        logger.info(f"Getting jet correction key {k}")
        corr = cset[k]
        # event_rho = getRho(events, jec_params.rho_name)
        factor = corr.evaluate(jets.eta, pt_raw)
        for shift_name, shift in [("up", 1), ("down", -1)]:
            # fields = {field: jets[field] for field in jets.fields}
            corrected = ak.with_field(jets, pt_raw * (1.0 + factor * shift), "pt")
            corrected = ak.with_field(
                corrected, mass_raw * (1.0 + factor * shift), "mass"
            )

            if do_smearing:
                corrected = smearJets(corrected, rho, params, cset, jet_type)

            systematic_name = f"{shift_name}_jes{systematic}"
            logger.info(f"Adding jet systematic {systematic_name}")

            systs[systematic_name] = corrected

    if do_smearing:
        rjets, systs_jer = smearJets(
            corrected,
            rho,
            params,
            cset,
            jet_type,
            include_systematics=True,
        )
    else:
        rjets = jets
        systs_jer = {}

    columns.add(output_col, rjets, systs | systs_jer | {"uncorrected": jets})


# @MODULE_REPO.register(ModuleType.Producer)
# def smeared_jets(
#     columns, params, input_col: str = None, output_col: str = None, jet_type="AK4"
# ):
#
#     corrections_path = jec_params.files[jet_type]
#     cset = correctionlib.CorrectionSet.from_file(corrections_path)
#
#     pt_raw = (1 - jets.rawFactor) * jets.pt
#     mass_raw = (1 - jets.rawFactor) * jets.mass
#
#     systs = {}
#     for systematic in systematics:
#         k = getKey(systematic, jet_type, params)
#         logger.info(f"Getting jet correction key {k}")
#         corr = cset[k]
#         # event_rho = getRho(events, jec_params.rho_name)
#         factor = corr.evaluate(jets.eta, pt_raw)
#         for shift_name, shift in [("up", 1), ("down", -1)]:
#             # fields = {field: jets[field] for field in jets.fields}
#             corrected = ak.with_field(jets, pt_raw * (1.0 + factor * shift), "pt")
#             corrected = ak.with_field(
#                 corrected, mass_raw * (1.0 + factor * shift), "mass"
#             )
#
#             systematic_name = f"{shift_name}_jes{systematic}"
#             logger.info(f"Adding jet systematic {systematic_name}")
#
#             systs[systematic_name] = corrected
#
#     columns.add(output_col, jets, systs)


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
