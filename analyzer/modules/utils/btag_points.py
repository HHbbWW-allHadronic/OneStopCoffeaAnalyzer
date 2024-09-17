import correctionlib
import logging
import functools as ft

logger = logging.getLogger(__name__)

@ft.cache
def getBTagCset(path):
    cset = correctionlib.CorrectionSet.from_file(path)
    return cset
    

def getBTagWP(params):
    cset = getBTagCset(params.btagging.path)
    ret=  {p: cset["deepJet_wp_values"].evaluate(p) for p in ("L", "M", "T")}
    logger.info(f"BTag Working Points for profile \"{profile.name}\" are:\n{ret}")
    return ret
