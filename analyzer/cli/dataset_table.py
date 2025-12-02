import itertools as it
import re
import io
import csv
from rich.table import Table

def createSampleTable(repo, pattern=None, as_csv=False):
    table = Table(title="Samples")
    table.add_column("Dataset")
    table.add_column("Sample Name")
    table.add_column("Number Events")
    table.add_column("Data/MC")
    table.add_column("Era")
    table.add_column("X-Sec")
    for dataset_name in sorted(repo):
        dataset = repo[dataset_name]
        for sample in dataset:
            xs = sample.x_sec
            table.add_row(
                dataset.name,
                sample.name,
                f"{str(sample.n_events)}",
                dataset.sample_type,
                f"{dataset.era}",
                f"{xs:0.3g}" if xs else "N/A",
            )
    if not as_csv:
        return table
    else:
        d = {x.header:x.cells for x in table.columns}
        output = output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        headers = list(d)
        vals = zip(*(d[x] for x in headers))
        writer.writerow(headers)
        for r in vals:
            writer.writerow(r)
        return output.getvalue()


def createDatasetTable(manager, pattern=None, as_csv=False):
    table = Table(title="Samples")
    table.add_column("Dataset")
    table.add_column("Num Samples")
    table.add_column("Data/MC")
    table.add_column("Era")
    everything = [manager[x] for x in sorted(manager)]
    for s in everything:
        table.add_row(
            s.name,
            f"{len(s)}",
            s.sample_type,
            f"{s.era}",
        )
    if not as_csv:
        return table
    else:
        d = {x.header:x.cells for x in table.columns}
        output = output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        headers = list(d)
        vals = zip(*(d[x] for x in headers))
        writer.writerow(headers)
        for r in vals:
            writer.writerow(r)
        return output.getvalue()
