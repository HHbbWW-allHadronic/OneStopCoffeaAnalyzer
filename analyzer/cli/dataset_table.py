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

def createPairDRTable(min_pair_idx, as_csv=False):
    pair_labels = [
        "b1-b2",
        "b1-q1", "b1-q2", "b1-q3", "b1-q4",
        "b2-q1", "b2-q2", "b2-q3", "b2-q4",
        "q1-q2",
        "q1-q3", "q1-q4",
        "q2-q3", "q2-q4",
        "q3-q4",
    ]

    counts = np.bincount(ak.to_numpy(min_pair_idx), minlength=15)
    total = len(min_pair_idx)
    sorted_pairs = sorted(
        zip(pair_labels, counts),
        key=lambda x: -x[1]
    )

    table = Table(title="Gen Quark Pair Min Delta R Frequency")
    table.add_column("Rank")
    table.add_column("Pair")
    table.add_column("Count")
    table.add_column("Frequency (%)")

    for rank, (label, count) in enumerate(sorted_pairs, start=1):
        table.add_row(
            str(rank),
            label,
            str(count),
            f"{100 * count / total:.1f}%",
        )

    if not as_csv:
        return table
    else:
        d = {x.header: x.cells for x in table.columns}
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        headers = list(d)
        vals = zip(*(d[x] for x in headers))
        writer.writerow(headers)
        for r in vals:
            writer.writerow(r)
        return output.getvalue()
