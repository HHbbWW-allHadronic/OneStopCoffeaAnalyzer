import subprocess
from rich import print
from collections import defaultdict
from pathlib import Path
import argparse
import subprocess


def getFiles():
    command = "eos root://cmseos.fnal.gov// find  -f '/store/user/ckapsiak/SingleStop/raw_official_samples/'"
    out = subprocess.run(command,shell=True, capture_output=True)
    lines = out.stdout.splitlines()
    lines = [l.decode("utf-8").replace("/eos/uscms/", "root://cmseos.fnal.gov//") for l in lines]
    return lines


def merge(output_file, input_files):
    print(f"Saving outfile {output_file} containing {len(input_files)} files")
    #output_file = Path(output_file)
    print(output_file)
    #output_file.parent.mkdir(exist_ok=True, parents=True)
    subprocess.run(["hadd", "-f", str(output_file), *input_files])


def mergeFiles(files, year_field, outdir):
    # files = Path(".").glob(input_pattern)
    # outdir = Path(outdir)
    year_sig = defaultdict(list)
    for f in files:
        fp = Path(f)
        # rel = f.relative_to(".")
        parts = fp.parts
        year = parts[year_field]
        sig = fp.name
        year_sig[(year, sig)].append(f)
    for (year, sig), f in year_sig.items():
        outfile =  outdir + "/"  + str(year) + "/" + sig
        merge(outfile, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year-field", type=int, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)

    args = parser.parse_args()

    mergeFiles(getFiles(),args.year_field, args.outdir)


if __name__ == "__main__":
    main()
