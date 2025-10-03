import itertools as it
import re
import io
import csv


from rich.table import Table


def createSampleTable(manager, pattern=None, as_csv=False):
    table = Table(title="Samples")
    table.add_column("Dataset")
    table.add_column("Sample Name")
    table.add_column("Number Events")
    table.add_column("Data/MC")
    table.add_column("Era")
    table.add_column("X-Sec")
    everything = list(
        it.chain.from_iterable((y.params for y in manager[x]) for x in sorted(manager))
    )
    everything.sort(key=lambda x: x.name)
    if pattern is not None:
        everything = [x for x in everything if pattern.match(x)]
    for s in everything:
        xs = s.x_sec
        table.add_row(
            s.dataset.name,
            s.name,
            f"{str(s.n_events)}",
            s.dataset.sample_type,
            f"{s.dataset.era.name}",
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
    if pattern is not None:
        everything = [x for x in everything if pattern.match(x.params)]
    for s in everything:
        table.add_row(
            s.params.name,
            f"{len(s)}",
            s.params.sample_type,
            f"{s.params.era.name}",
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

