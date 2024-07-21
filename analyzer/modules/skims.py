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
import sys
from uproot.writing._dask_write import ak_to_root


def isRootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    print(type(t))
    if isinstance(t, ak.types.NumpyType):
        return True
    if isinstance(t, ak.types.ListType) and isinstance(t.content, ak.types.NumpyType):
        return True
    return False


# def uprootWriteable(events):
#    """Restrict to columns that uproot can write compactly"""
#    # out_event = events[list(x for x in events.fields if not events[x].fields)]
#    out_event = events
#    for bname in events.fields:
#        if events[bname].fields:
#            out_event[bname] = ak.zip(
#                {
#                    n: ak.without_parameters(events[bname][n])
#                    for n in events[bname].fields
#                    if isRootcompat(events[bname][n])
#                }
#            )
#
#    return out_event


def uprootWriteable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    print(type(events))
    print(ak.backend(events))
    for bname in events.fields:
        if events[bname].fields:
            for x in events[bname].fields:
                print(f"{x} -- {isRootcompat(events[bname][x])}")
            

            out[bname] = ak.zip(
                {
                    n: ak.to_packed(ak.without_parameters(ak.to_regular(ak.to_regular(events[bname][n]))))
                    for n in events[bname].fields
                    if isRootcompat(ak.to_regular(events[bname][n]))
                }
            )
        else:
            out[bname] = ak.to_packed(ak.without_parameters(events[bname]))
        print("=" * 20)
    return out


def filterColumns(events, save_fields=None):
    print(save_fields)
    if save_fields:
        skimmed_dropped = events[
            list(set(x for x in events.fields if x in save_fields))
        ]
        print(skimmed_dropped.fields)
    else:
        skimmed_dropped = events

    return skimmed_dropped


@dask.delayed
def copyAndDeleteRootFiles(dep, directory, target):
    p = Path(directory).absolute()
    rfiles = list(p.rglob("*.root"))
    for f in rfiles:
        t = f"{target}/{str(f.relative_to(p.parent))}"
        copyFile(str(f), t)
    shutil.rmtree(p.parent)
    return None


def helloWorld(x, dataset_name):
    if ak.backend(x) == "typetracer":
        touched = ak.Array(
            ak.typetracer.length_zero_if_typetracer(x).layout.to_typetracer(
                forget_length=True
            )
        )
        touched = ak.Array(x.layout.to_typetracer(forget_length=True))
        return touched
    else:
        print("TYPE OF")
        f = uuid.uuid4()
        print(type(x))
        print(x.fields)
        print(ak.backend(x))
        print("==============")
        #ak_to_root(
        #    f"localfiles/x/{f}.root",
        #    x,
        #    "Events",
        #    "zlib",
        #    1,
        #    "",
        #    lambda counted: "n" + counted,
        #    lambda outer, inner: inner if outer == "" else outer + "_" + inner,
        #    10,
        #    10.0,
        #    None,
        #)

        with uproot.recreate(f"localfiles/x/{f}.root") as fout:
           fout["Events"] = uprootWriteable(x)
        return None


@analyzerModule("skim", categories="final")
def saveSkim(events, analyzer):
    if not analyzer.skim_save_path:
        raise ValueError(f"Must set skim save path to skim.")

    skimmed = filterColumns(events, list(reversed(analyzer.skim_save_cols)))
    # skimmed = uprootWriteable(skimmed)
    # dest = f"localfiles/{analyzer.dataset_name}/"
    ## skimmed = skimmed.repartition(rows_per_partition=150000)
    n_to_one_desired = 100
    np = skimmed.npartitions
    n_to_one = min(n_to_one_desired, np)
    # skimmed = skimmed.repartition(n_to_one=n_to_one)
    # dw = uproot.dask_write(
    #    skimmed,
    #    compute=False,
    #    destination=dest,
    #    tree_name="Events",
    #    prefix=f"{analyzer.dataset_name}",
    # ).to_delayed()

    print(type(skimmed))
    # skimmed = skimmed.compute()
    y = dak.map_partitions(
        helloWorld, skimmed, analyzer.dataset_name, meta=skimmed._meta
    )
    print("y is ")
    print(y)
    print("----------")

    # parq = dak.to_parquet(
    #    skimmed,
    #    f"localfiles/{analyzer.dataset_name}",
    #    prefix=f"{analyzer.dataset_name}",
    #    compute=False,
    # )

    analyzer.side_effect_computes = [
        y
        # parq.to_delayed(),
        # copyAndDeleteRootFiles(dw, dest, analyzer.skim_save_path),
    ]
    return events, analyzer
