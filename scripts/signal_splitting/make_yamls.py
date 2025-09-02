import subprocess
import uproot
from rich import print
from collections import defaultdict
from pathlib import Path
import argparse
import subprocess
import re
import csv
import yaml


xsec_dict = None


def getXS(year, sample_name):
    global xsec_dict
    if xsec_dict is None:
        xsec_dict = {}
        with open(Path(__file__).parent / "xsecs.csv") as f:
            reader = csv.reader(f, delimiter=" ")
            next(reader)
            for r in reader:
                coupling, mt, mx, run, sign, xsec, _ = r
                xsec_dict[(run, f"signal_{coupling}_{mt}_{mx}_{sign}")] = round(
                    float(xsec) * 1000 * 0.1**2, 4
                )
    run = "Run2" if year.startswith("201") else "Run3"
    return xsec_dict[(run, sample_name)]


def makeFile(fname, year, sample_name, base_name):
    with uproot.open(fname) as f:
        nevents = f["Events"].num_entries
    ret = {
        "name": sample_name,
        "n_events": nevents,
        "x_sec": getXS(year, base_name),
        "files": [fname],
    }
    return ret


def makeEntry(signal_name, year, plus, minus):
    coupling, mt, mx = re.search("signal_(31.)_([0-9]+)_([0-9]+)", signal_name).groups()
    name = f"signal_{year}_{coupling}_{mt}_{mx}_official"
    basename = f"signal_{coupling}_{mt}_{mx}"
    title = f"$m_{{\\tilde{{t}} }} = {mt}\\ \\mathrm{{GeV}}, m_{{\\tilde{{\\chi}}^{{\\pm}}}} = {mx}\\ \\mathrm{{GeV}}$"
    other_data = {"stop_mass": int(mt), "chargino_mass": int(mx), "coupling": coupling}
    ret = {
        "name": name,
        "title": title,
        "sample_type": "MC",
        "era": year,
        "other_data": other_data,
        "samples": [],
    }
    ret["samples"].append(makeFile(plus, year, name + "_plus", basename + "_plus"))
    ret["samples"].append(makeFile(minus, year, name + "_minus", basename + "_minus"))
    return ret


def getNameSign(fname):
    return re.search("(signal_31._[0-9]+_[0-9]+)_(plus|minus)", fname).groups()[0:2]


def getFiles(path):
    command = f"eos root://cmseos.fnal.gov// find  -f '{path}'"
    out = subprocess.run(command, shell=True, capture_output=True)
    lines = out.stdout.splitlines()
    lines = [
        l.decode("utf-8").replace("/eos/uscms/", "root://cmseos.fnal.gov//")
        for l in lines
    ]
    return lines


def makeGroups(files):
    ret = defaultdict(lambda: [None, None])
    for f in files:
        name, sign = getNameSign(f)
        ret[name][bool(sign == "minus")] = f
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True, parents=True)


    f = getFiles(args.input)
    g = makeGroups(f)
    g = {k: v for k, v in g.items() if not [x for x in v if x is None]}
    ret = []
    for group, items in g.items():
        e = makeEntry(group, args.year, *items)
        print(e)
        ret.append(e)

    yaml_string = yaml.dump(ret,sort_keys=False)
    with open(out,'w') as f:
        f.write(yaml_string)


if __name__ == "__main__":
    main()
