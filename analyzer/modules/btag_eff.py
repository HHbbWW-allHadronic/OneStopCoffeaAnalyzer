import awkward as ak
from analyzer.core import analyzerModule
import functools
from analyzer.configuration import getConfiguration
from pathlib import Path

import operator as op
import correctionlib
import correctionlib.convert
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from correctionlib.convert import from_histogram
import pickle as pkl
from .utils import isMC
from hist import axis
from .btag_points import getBTagWP


@analyzerModule(
    "btag_efficiencies",
    categories="preselection",
    depends_on=["objects"],
    dataset_pred=isMC,
)
def btagEffs(events, analyzer):
    profile = analyzer.profile
    bwps = getBTagWP(profile)

    config = getConfiguration()
    pt_bins = [30, 70, 100, 140, 200, 300, 600, 1000]
    pt_axis = axis.Variable(pt_bins, label="pt")
    eta_axis = axis.Regular(8, 0, 2.4, label="eta")
    gj = events.good_jets

    for wp, value in bwps.items():
        truth = ak.flatten(gj[gj.hadronFlavour == 5])
        tagged = ak.flatten(gj[gj.btagDeepFlavB >= value])
        analyzer.H(f"btag_{wp}_truth", [pt_axis, eta_axis], [truth.pt, abs(truth.eta)])
        analyzer.H(f"btag_{wp}_tagged", [pt_axis, eta_axis], [tagged.pt, abs(tagged.eta)])

    return events, analyzer
