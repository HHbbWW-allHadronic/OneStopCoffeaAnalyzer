import inspect
import logging


def getSectorLogger(params):
    fname = inspect.stack()[1][3]
    dataset_name = params.dataset.name
    region_name = params.region_name
    ret = logging.getLogger(f"analyzer.modules.{fname}")
    return ret
