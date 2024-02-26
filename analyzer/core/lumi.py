from urllib import parse
import coffea.lumi_tools as ltools
from pathlib import Path
import requests


def getLumiMask(lumi_json):
    scheme, netloc, path, *rest = parse.urlparse(lumi_json)
    netpath = Path(path)
    desired_fname = netpath.name
    lumi_data = Path("analyzer_resources") / "lumi_json"
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
