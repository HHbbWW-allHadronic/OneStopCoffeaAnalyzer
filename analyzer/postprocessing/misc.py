import csv
from pathlib import Path


def dumpYield(packaged_hists, output):
    output = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    with open(output, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for packaged_hist in packaged_hists:
            title = packaged_hist.title
            s = packaged_hist.histogram.sum()
            val = s.value
            var = s.variance
            writer.writerow([title, val, var])
