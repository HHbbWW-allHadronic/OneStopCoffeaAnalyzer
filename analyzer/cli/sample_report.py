import itertools as it
import re


from rich.table import Table


def createSampleTable(manager, re_filter=None):
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
    if re_filter:
        p = re.compile(re_filter)
        everything = [x for x in everything if p.search(x.name)]
    for s in everything:
        xs = s.x_sec
        table.add_row(
            s.dataset.name,
            s.name,
            f"{str(s.n_events)}",
            s.dataset.sample_type,
            f"{s.dataset.era}",
            f"{xs:0.2g}" if xs else "N/A",
        )
    return table
