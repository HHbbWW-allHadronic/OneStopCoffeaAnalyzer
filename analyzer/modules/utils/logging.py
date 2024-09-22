import inspect
import logging


def getSectorLogger(params):
    fname = inspect.stack()[1][3]
    sector_id = params["sector_id"].serialize()
    ret = logging.getLogger(f"analyzer.modules.{fname}.{sector_id}")
    return ret
