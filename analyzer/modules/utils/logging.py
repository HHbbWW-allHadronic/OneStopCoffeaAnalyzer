import inspect
import logging


def getSectorLogger(params):
    fname = inspect.stack()[1][3]
    ret = logging.getLogger(f"analyzer.modules.{fname}")
    return ret
