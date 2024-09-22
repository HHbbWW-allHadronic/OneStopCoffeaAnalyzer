import logging
from functools import cache
from pathlib import Path
from urllib import parse

import coffea.lumi_tools as ltools
import requests
from analyzer.configuration import CONFIG
from analyzer.core import MODULE_REPO, ModuleType

logger = logging.getLogger(__name__)


@cache
def getLumiMask(lumi_json):
    scheme, netloc, path, *rest = parse.urlparse(lumi_json)
    netpath = Path(path)
    desired_fname = netpath.name
    lumi_data = Path(CONFIG.APPLICATION_DATA) / "lumi_json"
    target_file = lumi_data / desired_fname
    if not target_file.is_file():
        logger.info(f"Json data file {target_file} does not exist.")
        lumi_data.mkdir(parents=True, exist_ok=True)
        logger.info(f'Fetching data from "{lumi_json}"')
        req = requests.get(lumi_json)
        with open(target_file, "wb") as f:
            f.write(req.content)
    lmask = ltools.LumiMask(target_file)
    return lmask


@MODULE_REPO.register(ModuleType.Selection)
def golden_lumi_filter(events, params, selector):
    lumi_json = params["golden_json"]
    logger.info(f"Applying lumi json {lumi_json}")
    lmask = getLumiMask(lumi_json)
    selector.add(f"golden_lumi", lmask(events.run, events.luminosityBlock), type="and")
