import argparse
import operator as op
import itertools as it
import datetime
import json
from rich.prompt import Prompt, Confirm
import logging
import subprocess
import sys
from pathlib import Path
from rich.progress import track
import re

import requests
import yaml
from pydantic import BaseModel, model_validator, TypeAdapter

from analyzer.configuration import CONFIG

from auth_get_sso_cookie import cern_sso
from rich import print


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


AUTH_HOSTNAME = "auth.cern.ch"
AUTH_REALM = "cern"


class ProtoDataset(BaseModel):
    name: str
    title: str
    cms_regex: str
    era: str
    sample_type: str
    filter_regex: str | None = None
    sample_field: int = 1
    locate_xsec: bool = False
    known_xsec: float | None = None
    process_field: int = 1

    @model_validator(mode="before")
    @classmethod
    def addLocate(cls, data):
        if data.get("sample_type") == "MC" and data.get("known_xsec") is None:
            data["locate_xsec"] = True
        return data

    @model_validator(mode="after")
    def checkX(self):
        if self.sample_type == "MC" and not (self.locate_xsec or self.known_xsec):
            raise ValueError("Cannot determine cross section for this sample")

        if self.locate_xsec and self.known_xsec:
            raise ValueError("Do not need to locate xsec if it is known.")

        return self

    @property
    def save_path(self):
        return (Path(self.era) / self.name).with_suffix(".yaml")


class XSecDBEntry(BaseModel):
    id: str
    process_name: str
    status: str
    cross_section: float
    total_uncertainty: float
    accuracy: str
    DAS: str
    MCM: str
    equivalent_lumi: float
    fraction_negative_weight: float
    shower: str
    matrix_generator: str
    energy: float
    comments: str
    modifiedOn: datetime.datetime
    createdOn: datetime.datetime
    modifiedBy: str
    createdBy: str


ProtoDatasetList = TypeAdapter(list[ProtoDataset])
XSecEntryList = TypeAdapter(list[XSecDBEntry])


def getToken():
    clientid = "public-client"
    # clientid = "single-stop-analyzer"
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


# def getSampleXSec(sample, token_file):
#     with open(token_file, "r") as f:
#         token = f.readline()
#
#     process = sample.split("/")[0]
#
#     url = "https://xsecdb-xsdb-official.app.cern.ch/api/search"
#
#     response = requests.post(
#         url,
#         json={"process_name": process},
#         headers={"Authorization": f"Bearer {token}"},
#     )
#     print(response.text)
#     req = response.request
#     print(
#         "{}\n{}\r\n{}\r\n\r\n{}".format(
#             "-----------START-----------",
#             req.method + " " + req.url,
#             "\r\n".join("{}: {}".format(k, v) for k, v in req.headers.items()),
#             req.body,
#         )
#     )
#     if not response:
#         return None
#     r = response.json()
#     return r


def loadAllXSecs():
    outfile = Path(CONFIG.APPLICATION_RESOURCES) / "misc" / "xsecs"
    files = outfile.rglob("*.json")
    data = []
    for f in files:
        with open(f, "r") as infile:
            data += XSecEntryList.validate_json(infile.read())
    k = op.attrgetter("process_name")
    grouped = it.groupby(sorted(data, key=k), key=k)
    ret = {x: sorted(y, key=op.attrgetter("total_uncertainty"))[0] for x, y in grouped}
    return ret


DAS_PATH = (
    "/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/dasgoclient/v02.04.52.rev00/bin/dasgoclient"
)


def query(query):
    dasgo_extra_args = ["--timeout", "3", "-noKeepAlive"]
    # dasgo_extra_args = []
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


def buildDatasetFromProto(protoset, output, xsec_db, skip_existing=True):
    output = Path(output)
    output = output / protoset.save_path

    if output.exists() and skip_existing:
        return

    ds = buildDataset(
        protoset.name,
        protoset.title,
        protoset.era,
        protoset.sample_type,
        protoset.cms_regex,
        xsec_db,
        include_xsec=protoset.locate_xsec,
        sample_name_field=protoset.sample_field,
        sample_process_field=protoset.process_field,
        filter_extra=protoset.filter_regex,
    )

    if ds:
        output.parent.mkdir(exist_ok=True, parents=True)
        with open(output, "w") as f:
            yaml.dump(
                ds,
                stream=f,
                sort_keys=False,
                Dumper=MyDumper,
                default_flow_style=False,
            )


def buildDataset(
    dataset_name,
    dataset_title,
    era,
    sample_type,
    query,
    xsec_database,
    include_xsec=True,
    token_file=None,
    sample_name_field=1,
    sample_process_field=2,
    filter_extra=None,
):
    header = {
        "name": dataset_name,
        "title": dataset_title,
        "sample_type": sample_type,
        "era": str(era),
        "samples": [],
    }
    samples = sorted(getSamples(query))

    if filter_extra:
        samples = [s for s in samples if re.search(filter_extra, s)]

    sample_info = [
        dict(query=sample, name=Path(sample).parts[sample_name_field])
        for sample in samples
    ]

    if sample_type == "MC":
        for d in sample_info:
            x = xsec_database.get(Path(d["query"]).parts[sample_process_field])
            if x is not None:
                d["x_sec"] = x.cross_section * 1000
    print(header)
    print(sample_info)

    ok = Confirm.ask("Are these samples Ok?")

    if not ok:
        return

    for sinfo in track(sample_info):
        name = sinfo["name"]
        sample = sinfo["query"]
        name = Path(sample).parts[sample_name_field]
        events = countEvents(sample)
        files = [f"root://cmsxrootd.fnal.gov/{x}" for x in getFiles(sample)]
        # x_sec = getSampleXSec(sample, token_file),
        cms_dataset_regex = sample
        sample_process = sample.split("/")[0]
        d = {
            "name": name,
            "n_events": events,
            "cms_dataset_regex": cms_dataset_regex,
        }

        x_sec = sinfo.get("x_sec")
        if x_sec is not None:
            d["x_sec"] = x_sec
        d["files"] = files
        header["samples"].append(d)

    return [header]


def getArgs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
    )
    parser.add_argument("input")
    parser.add_argument("-l", "--limit-regex")
    args = parser.parse_args()
    return args


def run(input_path, output_dir, limit_regex=None):
    # token = getToken()
    f = None
    with open(input_path) as f:
        data = yaml.safe_load(f)

    data = ProtoDatasetList.validate_python(data)
    if limit_regex:
        data = [x for x in data if re.search(limit_regex, x.name)]

    xsecs = loadAllXSecs()

    for d in data:
        try:
            buildDatasetFromProto(d, output_dir, xsecs)
        except KeyboardInterrupt as e:
            ok = Confirm.ask("Want to continue with next sample")
            if not ok:
                raise e
