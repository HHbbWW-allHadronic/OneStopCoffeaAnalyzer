import functools as ft
import logging

import correctionlib

logger = logging.getLogger("analyzer")


@ft.cache
def getBTagCset(path):
    cset = correctionlib.CorrectionSet.from_file(path)
    return cset


def getBTagWP(params):
    era_info = params.dataset.era
    cset = getBTagCset(era_info.btag_scale_factors["file"])
    ret = {p: cset["deepJet_wp_values"].evaluate(p) for p in ("L", "M", "T")}
    return ret
