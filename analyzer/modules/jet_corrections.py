import copy
from rich import print
import numpy as np
from analyzer.core import MODULE_REPO, ModuleType
from analyzer.utils.debugging import jumpIn


import awkward as ak
import correctionlib
import pydantic as pyd
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import correctionlib.schemav2 as cs
from functools import lru_cache

import logging

logger = logging.getLogger(__name__)


class JerConf(pyd.BaseModel):
    smear_path: str
    dr_min: float
    pt_min: float
    gen_jet_name: str
    gen_jet_idx_name: str


def to_f32(x):
    # return x
    return ak.values_astype(x, np.float32)


@lru_cache
def getCset(path):
    cset = correctionlib.CorrectionSet.from_file(path)
    return cset


@lru_cache
def getEvaluator(path, key):
    cset = getCset(path)
    return cset[key]


@MODULE_REPO.register(ModuleType.Selection)
def jet_veto_maps(events, params, selection, veto_type="jetvetomap"):
    veto_params = params.dataset.era.jet_veto_maps
    fname = veto_params.file
    name = veto_params.name
    cset = correctionlib.CorrectionSet.from_file(fname)
    eval_veto = cset[name]
    j = events.Jet
    j = j[(j.pt > 15) & ((j.jetId & 0b100) != 0) & ((j.chHEF + j.neEmEF) < 0.9)]
    vetoes = eval_veto.evaluate(veto_type, j.eta, j.phi)
    selection.add("jet_veto_map", ak.any(vetoes, axis=1))


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


def smearJets(jets, gen_jets, rho, params, jet_type, include_systematics=False):
    jec_params = params.dataset.era.jet_corrections
    systematics = jec_params.jer.systematics
    evaljer = getEvaluator(
        jec_params.files[jet_type], getKeyJer("PtResolution", jet_type, params)
    )
    evalsf = getEvaluator(
        jec_params.files[jet_type], getKeyJer("ScaleFactor", jet_type, params)
    )
    evalsmear = getEvaluator(jec_params.files["smear"], "JERSmear")

    genjet_idx_col = jec_params.jer.genjet_idx_col

    inputs = {
        "JetEta": jets.eta,
        "JetPt": jets.pt,
        "Rho": rho,
    }
    jer = evaljer.evaluate(*[inputs[inp.name] for inp in evaljer.inputs])
    jer = to_f32(jer)

    def smearWithSystematic(jets, systematic_name):
        sf_inputs = inputs | {"systematic": systematic_name}
        sf = evalsf.evaluate(*[sf_inputs[inp.name] for inp in evalsf.inputs])
        sf = to_f32(sf)

        matched_genjet = jets.matched_gen

        gen_idxs = jets[genjet_idx_col]
        valid_gen_idxs = ak.mask(gen_idxs, (gen_idxs >= 0) & (gen_idxs < 10))
        padded_gen_jets = ak.pad_none(gen_jets, 10, axis=1)
        matched_genjet = padded_gen_jets[valid_gen_idxs]

        pt_relative_diff = 1 - matched_genjet.pt / jets.pt
        is_matched_pt = abs(pt_relative_diff) < (3 * jer)
        # is_matched_pt = ak.fill_none(is_matched_pt, False)

        # sf_scaling = 1.0 + (sf - 1.0) * pt_relative_diff

        smear_inputs = inputs | {
            "JER": jer,
            "JERSF": sf,
            "GenPt": ak.fill_none(is_matched_pt * matched_genjet.pt, -1),
            "EventID": ak.values_astype(rho * 10**6, "int64"),
        }

        final_smear = evalsmear.evaluate(
            *[smear_inputs[inp.name] for inp in evalsmear.inputs]
        )
        # jumpIn(**locals())
        final_smear = final_smear
        final_smear = to_f32(final_smear)

        smeared_jets = ak.with_field(jets, to_f32(jets.pt * final_smear), "pt")
        smeared_jets = ak.with_field(
            smeared_jets, to_f32(jets.mass * final_smear), "mass"
        )
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
    include_systematics=True,
    use_regrouped=True,
):
    jec_params = params.dataset.era.jet_corrections

    if use_regrouped:
        systematics = jec_params.jec.regrouped_systematics
    else:
        systematics = jec_params.jec.systematics

    jets = columns[input_col]

    corrections_path = jec_params.files[jet_type]
    # cset = correctionlib.CorrectionSet.from_file(corrections_path)

    pt_raw = (1 - jets.rawFactor) * jets.pt
    mass_raw = (1 - jets.rawFactor) * jets.mass
    rho = (
        columns.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in columns.fields
        else columns.Rho.fixedGridRhoFastjetAll
    )

    systs = {}

    if include_systematics:
        for systematic in systematics:
            k = getKeyJec(systematic, jet_type, params)
            logger.info(f"Getting jet correction key {k}")
            corr = getEvaluator(corrections_path, k)
            # event_rho = getRho(events, jec_params.rho_name)
            factor = to_f32(corr.evaluate(jets.eta, pt_raw))
            for shift_name, shift in [("up", 1), ("down", -1)]:
                # fields = {field: jets[field] for field in jets.fields}

                corrected = jets
                # corrected["pt"] = jets.pt * (1.0 + factor * shift)
                # corrected["mass"] = jets.mass * (1.0 + factor * shift)
                corrected = ak.with_field(
                    corrected, to_f32(jets.pt * (1.0 + factor * shift)), "pt"
                )
                corrected = ak.with_field(
                    corrected, to_f32(jets.mass * (1.0 + factor * shift)), "mass"
                )

                if do_smearing:
                    corrected = smearJets(
                        corrected, columns.GenJet, rho, params, jet_type
                    )

                systematic_name = f"{shift_name}_jes{systematic}"
                logger.info(f"Adding jet systematic {systematic_name}")

                systs[systematic_name] = corrected

    if do_smearing:
        rjets = smearJets(
            jets,
            columns.GenJet,
            rho,
            params,
            jet_type,
            include_systematics=include_systematics,
        )
        systs_jer = {}
        if include_systematics:
            rjets, systs_jer = rjets
    else:
        rjets = jets
        systs_jer = {}

    columns.add(output_col, rjets, systs | systs_jer | {"base": jets})
