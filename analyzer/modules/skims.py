import uuid

import awkward as ak
import dask
import dask_awkward as dak
import uproot
from analyzer.core import analyzerModule
from analyzer.file_utils import copyFile
from coffea.dataset_tools import apply_to_fileset, preprocess
from coffea.nanoevents import NanoAODSchema
from dask.distributed import get_worker
from dask.distributed import print as dprint
from pathlib import Path
import shutil
import fsspec
import fsspec_xrootd as fsx


def isRootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = dak.type(a)
    if isinstance(t, ak.types.NumpyType):
        return True
    if isinstance(t, ak.types.ListType) and isinstance(t.content, ak.types.NumpyType):
        return True
    return False


def uprootWriteable(events):
    """Restrict to columns that uproot can write compactly"""
    out_event = events[list(x for x in events.fields if not events[x].fields)]
    for bname in events.fields:
        if events[bname].fields:
            out_event[bname] = ak.zip(
                {
                    n: ak.without_parameters(events[bname][n])
                    for n in events[bname].fields
                    if isRootcompat(events[bname][n])
                }
            )
    return out_event


def filterColumns(events, save_fields=None):
    if save_fields:
        skimmed_dropped = events[
            list(set(x for x in events.fields if x in save_fields))
        ]
    else:
        skimmed_dropped = events

    return skimmed_dropped


@dask.delayed
def copyAndDeleteRootFiles(dep, directory, target):
    p = Path(directory)
    rfiles = list(p.rglob("*.root"))
    for f in rfiles:
        t = f"{target}/{str(f.relative_to(p.parent))}"
        copyFile(str(f), t)
    shutil.rmtree(p.parent)
    return None


@analyzerModule("skim", categories="final")
def saveSkim(events, analyzer):
    if not analyzer.skim_save_path:
        raise ValueError(f"Must set skim save path to skim.")
    skimmed = filterColumns(events, analyzer.skim_save_cols)
    #skimmed = filterColumns(events, None)
    print(skimmed.fields)
    skimmed = uprootWriteable(skimmed)
    #skimmed = skimmed.repartition(n_to_one=1_000)
    dest = f"localfiles/{uuid.uuid4()}/{analyzer.dataset_name}"
    dw = uproot.dask_write(
        skimmed,
        compute=True,
        destination=dest,
        prefix=f"{analyzer.dataset_name}",
    ).to_delayed()
    analyzer.side_effect_computes = [
        dw,
        copyAndDeleteRootFiles(dw, dest, analyzer.skim_save_path),
    ]
    return events, analyzer
