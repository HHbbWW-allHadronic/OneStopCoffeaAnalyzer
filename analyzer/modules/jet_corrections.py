import copy

import numpy as np

import awkward as ak
import correctionlib
import pydantic as pyd
from coffea.jetmet_tools import CorrectedMETFactory
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import correctionlib.schemav2 as cs


def addRaw(jets, event_rho, isMC=True):
    # if isMC:
    #     jets["pt_gen"] = ak.values_astype(
    #         ak.fill_none(jets.matched_gen.pt, 0), np.float32
    #     )
    return jets


def getCorrKey(base_tag, lvl, algo):
    return "_".join([base_tag, lvl, algo])


# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py
# https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/lib/jets.py
def jetsJEC(jets, base_tag, lvl, algo, corrections_path):
    cset = correctionlib.CorrectionSet.from_file(corrections_path)
    corr_key = getCorrKey(base_tag, lvl, algo)
    corr = correctionlib_wrapper(cset.compound[corr_key])

    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]

    factor = corr.evaluate(jets["area"], jets["eta"], jets["pt_raw"], jets["rho"])
    corrected = ak.copy(jets)
    corrected["rho"] = jets["rho"]
    corrected["pt"] = jets["pt_raw"] * factor
    corrected["mass"] = jets["mass_raw"] * factor

    return corrected


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
    cset_jersmear = core.CorrectionSet.from_file(jer_conf.smear_path)
    sf_jersmear =  correctionlib_wrapper(cset_jetsmear[key_jersmear])
    
    sf_key, res_key = getJerKeys(jer, jet_type)
    sf_corr = correctionlib_wrapper(cset[sf_key])
    res_corr = correctionlib_wrapper(cset[res_key])
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

    jer_smear_factor = sf_jersmear.evaluate(matched_jets.pt, matched_jets.eta, matched_jets.rho, events.event, sf, res)
    rand = resrng.evaluate(jets.pt. jets.eta, events.event)
    sqed = sf**2 -1
    rand_smear = 1 + (rand * res) * ak.sqrt(ak.where(sqed >0, sqed, 0))
    is_matched = ~ak.is_none(matched_jets.pt, axis=1)
    final_factor = ak.where(is_matched, jer_smear_factor, rand_smear)
    jets_smeared = copy.copy(jets)
    jets_smeared['pt'] = jets['pt'] * final_factor
    jets_smeared['mass'] = jets['mass'] * final_factor
    return jets_smeared




