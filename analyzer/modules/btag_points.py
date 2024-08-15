import correctionlib
import logging


logger = logging.getLogger(__name__)

cache = {}
def getBTagWP(profile):
    global cache
    n = profile.name
    if n in cache:
        return cache[n]
    cset = correctionlib.CorrectionSet.from_file(profile.btag_scale_factors)
    ret=  {p: cset["deepJet_wp_values"].evaluate(p) for p in ("L", "M", "T")}
    cache[n] = ret
    logger.info(f"BTag Working Points for profile \"{profile.name}\" are:\n{ret}")
    return ret
