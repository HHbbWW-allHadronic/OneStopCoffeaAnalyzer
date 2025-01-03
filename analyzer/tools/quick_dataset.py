import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path
import re

import requests
import yaml

from analyzer.configuration import CONFIG

# from auth_get_sso_cookie import cern_sso
from rich import print


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


AUTH_HOSTNAME = "auth.cern.ch"
AUTH_REALM = "cern"


def getToken():
    clientid = "public-client"
    clientid = "single-stop-analyzer"
    # clientid = "xsdb-official"#single-stop-analyzer"
    nocertify = False
    auth_server = "auth.cern.ch"
    auth_realm = "cern"

    outfile = Path(CONFIG.APPLICATION_DATA) / "token.txt"

    if outfile.exists():
        mod_time = datetime.datetime.fromtimestamp(outfile.lstat().st_mtime)
        now = datetime.datetime.now()
        diff = now - mod_time
        if diff < datetime.timedelta(minutes=18):
            return outfile

    try:
        token = cern_sso.device_authorization_login(
            clientid, not nocertify, auth_server, auth_realm
        )
        # if args.audience:
        #     token = cern_sso.public_token_exchange(
        #         clientid,
        #         audience,
        #         token["access_token"],
        #         auth_server,
        #         auth_realm,
        #         not nocertverify,
        #     )
        with open(outfile, "w", encoding="utf-8") as file:
            file.write(token["access_token"])
    except Exception as exception:
        logging.error(exception)
        sys.exit(1)
    return outfile


def getSampleXSec(sample, token_file):
    with open(token_file, "r") as f:
        token = f.readline()

    process = sample.split("/")[0]

    url = "https://xsecdb-xsdb-official.app.cern.ch/api/search"

    response = requests.post(
        url, json={"process_name": process}, headers={"Authorization": "Bearer {token}"}
    )
    print(response.text)
    req = response.request
    print(
        "{}\n{}\r\n{}\r\n\r\n{}".format(
            "-----------START-----------",
            req.method + " " + req.url,
            "\r\n".join("{}: {}".format(k, v) for k, v in req.headers.items()),
            req.body,
        )
    )
    if not response:
        return None
    r = response.json()
    return r


DAS_PATH = (
    "/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/dasgoclient/v02.04.52.rev00/bin/dasgoclient"
)


def query(query):
    dasgo_extra_args = ["--timeout", "3", "-noKeepAlive"]
    p = subprocess.Popen(
        [DAS_PATH, *dasgo_extra_args, "--json", "--query", query],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # stdout, stderr = p.communicate()
    o = []
    for l in p.stdout:
        o.append(l.decode())
        # print(l.decode())
    p.stdout.close()
    rc = p.wait()
    stdout = "\n".join(o)
    # print(stdout)
    if not rc:
        return json.loads(stdout)
    raise RuntimeError(f"Failed to query DAS:\n{stdout}")


def countEvents(sample):
    q = query(f"file dataset={sample} | sum(file.nevents)")
    return q[0]["result"]["value"]


def getFiles(sample):
    q = query(f"file dataset={sample}")
    return [x["file"][0]["name"] for x in q]


def getSamples(dataset):
    return sorted([x["dataset"][0]["name"] for x in query(f"dataset={dataset}")])


def buildDataset(
    dataset_name,
    dataset_title,
    era,
    sample_type,
    query,
    include_xsec=True,
    token_file=None,
    sample_name_field=1,
    filter_extra=None,
):
    header = {
        "name": dataset_name,
        "title": dataset_title,
        "sample_type": sample_type,
        "era": str(era),
        "samples": [],
    }
    samples = getSamples(query)
    if filter_extra:
        samples = [s for s in samples if re.search(filter_extra, s)]
    print(samples)
    yn = input("Ok? (y/n)")
    if yn != "y":
        sys.exit()
    for sample in samples:
        print(sample)
        name = Path(sample).parts[sample_name_field]
        print(name)
        events = countEvents(sample)
        files = [f"root://cmsxrootd.fnal.gov/{x}" for x in getFiles(sample)]
        x_sec = None
        # x_sec = getSampleXSec(sample, token_file),
        cms_dataset_regex = sample
        sample_process = sample.split("/")[0]
        d = {
            "name": name,
            "n_events": events,
            "cms_dataset_regex": cms_dataset_regex,
            "files": files,
        }
        if include_xsec:
            d["x_sec"] = x_sec
        header["samples"].append(d)

    return [header]


def getArgs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--title",
        required=True,
    )
    parser.add_argument(
        "-y",
        "--era",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--type",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--filter-extra",
        help="Filter extra",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--sample-name-field",
        required=True,
        type=int,
    )
    parser.add_argument("-x", "--no-include-xsec", default=False, action="store_true")
    parser.add_argument(
        "query",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    f = None
    args = getArgs()
    q = args.query
    output = args.output
    ds = buildDataset(
        args.name,
        args.title,
        args.era,
        args.type,
        q,
        include_xsec=not args.no_include_xsec,
        token_file=f,
        sample_name_field=args.sample_name_field,
        filter_extra=args.filter_extra,
    )
    with open(output, "w") as f:
        yaml.dump(
            ds,
            stream=f,
            sort_keys=False,
            Dumper=MyDumper,
            default_flow_style=False,
        )
