import http.cookiejar
import logging
from pathlib import Path
from rich import print
import sys

import requests

from analyzer.configuration import CONFIG
from auth_get_sso_cookie import cern_sso


def getCookie():
    outfile = Path(CONFIG.APPLICATION_DATA) / "cookie.txt"
    url = "https://xsecdb-xsdb-official.app.cern.ch/api/search"
    try:
        if outfile.exists():
            with open(outfile, "r", encoding="utf-8") as file:
                try:
                    keycloak_sessions = [
                        line.split("\t")
                        for line in file.readlines()
                        if "KEYCLOAK_SESSION" in line
                    ]
                    current_ts = datetime.now() - timedelta(minutes=10)
                    expire_ts = datetime.utcfromtimestamp(int(keycloak_sessions[0][4]))
                    if expire_ts > current_ts:
                        logging.warning(
                            "The existing cookie in file '%s' is still valid until %s. "
                            + "This run will not start a new session, please use the existing file.",
                            outfile,
                            expire_ts,
                        )
                        return
                except:
                    pass
        cern_sso.save_sso_cookie(url, outfile, True, "auth.cern.ch")
    except Exception as exception:
        logging.error(exception)
        sys.exit(1)
    return outfile


def getSampleXSec(sample_name, cookie):
    url = "https://xsecdb-xsdb-official.app.cern.ch/api/search"
    response = requests.post(url, cookies=cookie, json={"process_name": sample_name})
    if not response:
        return None
    r = response.json()
    print(r)


def main():
    from analyzer.datasets.samples import DatasetRepo

    m = http.cookiejar.MozillaCookieJar(getCookie())
    m.load()
    repo = DatasetRepo.getConfig()
    for dataset in repo:
        ds = repo[dataset]
        for sample in ds.samples:
            if not sample.cms_dataset_regex:
                continue
            n = sample.cms_dataset_regex.split("/")[1]
            x = getSampleXSec(n, m)


if __name__ == "__main__":
    main()
