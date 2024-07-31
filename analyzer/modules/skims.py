import uuid

import awkward as ak
import dask
import dask_awkward as dak
import uproot
from analyzer.core import analyzerModule
import analyzer.file_utils as fu
from coffea.dataset_tools import apply_to_fileset, preprocess
from coffea.nanoevents import NanoAODSchema
from dask.distributed import get_worker
from pathlib import Path
import shutil
import fsspec
import fsspec_xrootd as fsx
import sys
from uproot.writing._dask_write import ak_to_root


def isRootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        if isinstance(t.content, ak.types.NumpyType):
            return True
        if isinstance(t.content, ak.types.ListType) and isinstance(
            t.content.content, ak.types.NumpyType
        ):
            return True
    return False


def uprootWriteable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in events.fields:
        if events[bname].fields:
            out[bname] = ak.zip(
                {
                    n: ak.without_parameters(events[bname][n])
                    for n in events[bname].fields
                    if isRootcompat(events[bname][n])
                }
            )
        else:
            out[bname] = ak.to_packed(ak.without_parameters(events[bname]))
    return out


def filterColumns(events, save_fields=None):
    if save_fields:
        skimmed_dropped = events[
            list(set(x for x in events.fields if x in save_fields))
        ]
    else:
        skimmed_dropped = events

    return skimmed_dropped


def skimmer(x, fname_prefix, dest, tempdir="localfiles/"):
    tempdir = Path(tempdir)
    tempdir.mkdir(exist_ok=True, parents=True)
    if ak.backend(x) == "typetracer":
        touched = ak.Array(
            ak.typetracer.length_zero_if_typetracer(x).layout.to_typetracer(
                forget_length=True
            )
        )
        return touched
    else:
        f = str(uuid.uuid4())
        fname = tempdir / f"{fname_prefix}{f}.root"
        with uproot.recreate(fname) as fout:
            fout["Events"] = uprootWriteable(x)
        fu.copyFile(fname, dest, from_rel_to=tempdir)
        Path(fname).unlink()
        return None
        #return fname


@analyzerModule("skim", categories="final")
def saveSkim(events, analyzer):
    if not analyzer.skim_save_path:
        raise ValueError(f"Must set skim save path to skim.")

    skimmed = filterColumns(events, list(reversed(analyzer.skim_save_cols)))
    n_to_one_desired = 100
    np = skimmed.npartitions
    n_to_one = min(n_to_one_desired, np)
    y = dak.map_partitions(
        skimmer,
        skimmed,
        f"{analyzer.dataset_name}/",
        analyzer.skim_save_path,
        meta=skimmed._meta,
    )
    analyzer.side_effect_computes = [
        y
        # parq.to_delayed(),
        # copyAndDeleteRootFiles(dw, dest, analyzer.skim_save_path),
    ]
    return events, analyzer
