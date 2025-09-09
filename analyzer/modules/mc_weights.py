import logging
import pickle as pkl
from pathlib import Path

import awkward as ak
import correctionlib
import correctionlib.convert
from analyzer.core import MODULE_REPO, ModuleType
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from correctionlib.convert import from_histogram
from analyzer.core.exceptions import AnalysisConfigurationError

from .utils.btag_points import getBTagWP


@MODULE_REPO.register(ModuleType.Weight)
def alpha_s(events, params, weight_manager):
    nom = ak.oneslike(events.run, axis=0)
    # names = {
    #     # ("down", "down"): 0,
    #     ("down", "nom"): 1,
    #     ("down", "up"): 2,
    #     ("nom", "down"): 3,
    #     # ("nom", "nom"): 4,
    #     ("nom", "up"): 5,
    #     ("up", "down"): 6,
    #     ("up", "nom"): 7,
    #     # ("up", "up"): 8,
    # }
    names = {"down": [0,1], "nom": [3, 5], "up": [7,8]}
    systs = {}
    for name, idxs in names.items():
        systs[name] = [events.LHEScaleWeight[i] for i in idxs]

    weight_manager.add(f"LHEScale", nom, systs)
