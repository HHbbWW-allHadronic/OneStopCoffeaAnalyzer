import logging
import pickle as pkl
from pathlib import Path

import awkward as ak
import correctionlib
import correctionlib.convert
from analyzer.core import MODULE_REPO, ModuleType
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from correctionlib.convert import from_histogram

from .utils.btag_points import getBTagWP

logger = logging.getLogger(__name__)


@MODULE_REPO.register(ModuleType.Weight)
def btagging_wp_sf(
    events, params, weight_manager, variations=None, working_points=None
):

    bwps = getBTagWP(profile)
    WP_ORDER = ("L", "M", "T")
    used_working_points = analyzer.processing_info["used_btag_wps"]
    wp_names = [x for x in WP_ORDER if x in used_working_points]

    efficiency_path = Path(config["PHYSICS_DATA"]) / "btag_efficiencies"
    target = efficiency_path / f"{analyzer.last_ancestor}.pkl"
    if not target.exists():
        raise RuntimeError(
            f'Computing btag scale factors requires knowing efficiencies of the pre-selected simulated sample.\nCould not find an efficiency histogram "{str(target)}".'
        )
    with open(target, "rb") as f:
        eff_data = pkl.load(f)
    efficiencies = {p: from_histogram(eff_data[p]) for p in wp_names}
    for x in efficiencies.values():
        x.data.flow = "clamp"

    gj = events.good_jets
    corrs = correctionlib.CorrectionSet.from_file(profile.btag_scale_factors)

    sf_eval = correctionlib_wrapper(corrs["deepJet_comb"])
    eff_eval = {
        p: correctionlib_wrapper(c.to_evaluator()) for p, c in efficiencies.items()
    }

    N = len(wp_names)

    # L -> T 1->N

    def getE(idx, jets):
        """Get the efficiency of working point index i. Here i=0 corresponds to 1 and i=N corresponds to 0"""
        if idx == 0:
            return ak.ones_like(jets.pt)
        elif idx == N + 1:
            return ak.zeros_like(jets.pt)
        else:
            return eff_eval[wp_names[idx - 1]](jets.pt, abs(jets.eta))

    def getMask(idx, jets):
        """Get jets passing working point wp but not wp-1, where wp-1 is the next tighter working point"""
        # idx = 0 is untagged
        if idx == 0:
            return jets.btagDeepFlavB < bwps[wp_names[-1]]
        # idx = N+1 is tagged only with the tightest working point
        elif idx == N + 1:
            return jets.btagDeepFlavB >= bwps[wp_names[idx - 1]]
        # else means tagged with wp N but not N+1
        else:
            return (jets.btagDeepFlavB >= bwps[wp_names[idx - 1]]) & (
                jets.btagDeepFlavB < bwps[wp_names[idx]]
            )

    def getSF(var, idx, jets):
        if idx == 0:
            return 1
        else:
            wp = wp_names[idx - 1]
            sf_passed = sf_eval(var, wp, 5, abs(jets.eta), jets.pt)
            return sf_passed

    def computeWeight(var, wp_names):
        p_mc, p_d = 1, 1
        for i, _ in enumerate(wp_names):
            m = getMask(i, gj)
            e1 = getE(i, gj[m])
            e2 = getE(i + 1, gj[m])

            s1 = getSF(var, i, gj[m])
            s2 = getSF(var, i + 1, gj[m])

            p_mc = p_mc * (ak.prod(e1, axis=1) - ak.prod(e2, axis=1))
            p_d = p_d * (ak.prod(s1 * e1, axis=1) - ak.prod(s2 * e2, axis=1))

        return p_d / p_mc

    s = {
        f"var": (
            computeWeight("up_correlated", wp_names),
            computeWeight("down_correlated", wp_names),
        )
    }
    weight_manager.add(f"btag_sf", computeWeight("central", wp_names), s)


@MODULE_REPO.register(ModuleType.Weight)
def pileup_sf(events, params, weight_manager, variations=None):
    return
    pu_info = params.dataset.era.pileup_scale_factors
    path = pu_info["file"]
    name = pu_info["name"]
    csset = correctionlib.CorrectionSet.from_file(path)
    logger.debug(f'Applying pu_sf from file "{path}" with name "{name}"')
    corr = csset[name]
    n_pu = events.Pileup.nTrueInt
    nom = corr.evaluate(n_pu, "nominal")
    up = corr.evaluate(n_pu, "up")
    down = corr.evaluate(n_pu, "down")
    logging.info(nom)
    logging.info(up)
    logging.info(down)
    weight_manager.add(f"pileup_sf", nom, {"inclusive": (up, down)})


@MODULE_REPO.register(ModuleType.Weight)
def L1_prefiring_sf(events, params, weight_manager, variations=None):
    nom = events.L1PreFiringWeight["Nom"]
    up = events.L1PreFiringWeight["Up"]
    down = events.L1PreFiringWeight["Dn"]
    weight_manager.add(f"L1_prefire", nom, {"inclusive": (up, down)})




@MODULE_REPO.register(ModuleType.Weight)
def btagging_shape_sf(
    events, params, weight_manager, variations=None, working_points=None
):
    current_syst = events.getSystName()

    jets = events.Jet
    corrs = correctionlib.CorrectionSet.from_file(profile.btag_scale_factors)

    sf_eval = correctionlib_wrapper(corrs["deepJet_shape"])
    central_perjet = sf_eval(
        "central", jets.hadronFlavour, abs(jets.eta), jets.pt, jets.btagDeepFlavB
    )

    if current_syst.starts_with("JES"):
        pass

    central = ak.prod(central_perjet, axis=1)

    weight_manager.add(
        f"btag_shape_sf",
        central,
    )
