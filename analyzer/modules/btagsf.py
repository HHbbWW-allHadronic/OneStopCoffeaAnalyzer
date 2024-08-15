import awkward as ak
from analyzer.core import analyzerModule
from .utils import isMC
import functools
from analyzer.configuration import getConfiguration
from pathlib import Path
import itertools as it

import operator as op
import correctionlib
import correctionlib.convert
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from correctionlib.convert import from_histogram
import pickle as pkl
from .btag_points import getBTagWP


@analyzerModule(
    "btag_scalefactors",
    categories="preselection",
    always=False,
    dataset_pred=isMC,
    depends_on=["objects"],
)
def btagScaleFactors(events, analyzer):
    config = getConfiguration()
    profile = analyzer.profile
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

    # for x in wp_names:
    s = {
        f"var": (
            computeWeight("up_correlated", wp_names),
            computeWeight("down_correlated", wp_names),
        )
    }
    analyzer.addWeight(f"btag_sf", computeWeight("central", wp_names), s)

    return events, analyzer
