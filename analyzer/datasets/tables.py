import itertools as it
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from yaml import dump, load

import rich
from coffea.dataset_tools.preprocess import DatasetSpec
from rich.table import Table

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


class ForbiddenDataset(Exception):
    pass



def createSetTable(manager, re_filter=None):
    table = Table(title="Samples Sets")
    table.add_column("Name")
    table.add_column("Number Events")
    table.add_column("Data/MC")
    table.add_column("X-Sec")
    table.add_column("Lumi")
    table.add_column("Number Files")
    table.add_column("Derived From")
    everything = list(manager.sets.values())
    if re_filter:
        p = re.compile(re_filter)
        everything = [x for x in everything if p.search(x.name)]
    for s in everything:
        xs = s.getXSec()
        lumi = s.getLumi()
        table.add_row(
            s.name,
            f"{str(s.totalEvents())}",
            "Data" if s.isData() else "MC",
            f"{xs:0.2g}" if xs else "N/A",
            f"{lumi:0.4g}" if lumi else "N/A",
            f"{len(s.files)}",
            f"{s.derived_from.name}" if s.derived_from else "N/A",
        )
    return table


def createCollectionTable(manager, re_filter=None):
    table = Table(title="Samples Collections")
    table.add_column("Name")
    table.add_column("Number Events")
    table.add_column("Number Sets")
    table.add_column("Treat Separate")
    everything = list(manager.collections.values())
    if re_filter:
        p = re.compile(re_filter)
        everything = [x for x in everything if p.search(x.name)]
    for s in everything:
        table.add_row(
            s.name, f"{str(s.totalEvents())}", f"{len(s.sets)}", f"{s.treat_separate}"
        )
    return table
