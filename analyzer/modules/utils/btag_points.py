import functools as ft
import logging

import correctionlib

logger = logging.getLogger(__name__)


@ft.cache
def getBTagCset(path):
    cset = correctionlib.CorrectionSet.from_file(path)
    return cset


def getBTagWP(params):
    return {"L": 0.7, "M": 0.8, "T": 0.9}
    era_info = params.sector.dataset.era
    cset = getBTagCset(era_info.btag_scale_factors["file"])
    ret = {p: cset["deepJet_wp_values"].evaluate(p) for p in ("L", "M", "T")}
    return ret
