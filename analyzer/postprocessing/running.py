import concurrent.futures as cf
import itertools as it

import matplotlib as mpl
from collections import defaultdict
from analyzer.core.results import (
    loadSampleResultFromPaths,
    makeDatasetResults,
    gatherFilesByPattern,
)
from rich import print
from rich.progress import Progress, track
from distributed import (
    Client,
    LocalCluster,
    fire_and_forget,
    get_client,
    secede,
    rejoin,
)

from .plots.mplstyles import loadStyles

# from .processors import PostProcessorType, postprocess_catalog
from .registry import loadPostprocessors

import analyzer.postprocessing.processors  # noqa
import analyzer.postprocessing.exporting  # noqa
import analyzer.postprocessing.basic_histograms  # noqa
import analyzer.postprocessing.cutflows  # noqa


def initProcess():
    mpl.use("Agg")
    loadStyles()


def run(tasks, parallel):
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Processing...", total=None)
        if not parallel:
            for f in tasks:
                f()
                progress.advance(task_id)
        else:
            with cf.ProcessPoolExecutor(
                max_workers=parallel, initializer=initProcess
            ) as executor:
                results = [executor.submit(f) for f in tasks]
                for i in cf.as_completed(results):
                    try:
                        i.result()
                    except Exception as e:
                        raise
                        print(f"Exception occurred {e}")
                    progress.advance(task_id)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def processFileGroup(files, postprocessors, use_samples_as_datasets=False, drops=None):
    all_needed_hists = [y for x in postprocessors for y in x.getNeededHistograms()]
    sample_results = loadSampleResultFromPaths(
        files, include=all_needed_hists, show_progress=False
    )

    def dropSampleFunction(sid):
        if drops is None:
            return False
        return drops.match(sid.sample_name)

    dataset_results = makeDatasetResults(
        sample_results,
        drop_sample_fn=dropSampleFunction,
        include_samples_as_datasets=use_samples_as_datasets,
    )
    sector_results = list(
        it.chain.from_iterable(r.sector_results for r in dataset_results.values())
    )
    ret = []
    client = get_client()
    for x in it.chain.from_iterable(
        (processor.getExe(sector_results) for processor in postprocessors)
    ):
        fire_and_forget(client.submit(x))


def assembleFileGroup(param_dict, fields):
    ret = defaultdict(set)
    for k, v in param_dict.items():
        newk = frozenset((x, y) for x, y in k if x in fields)
        ret[newk] |= v
    return ret


def runPostprocessors(config, input_files, parallel=8):
    loadStyles()
    print("Loading Postprocessors")
    loaded, catalog, drops, use_samples_as_datasets, num_files = loadPostprocessors(
        config
    )
    for processor in loaded:
        processor.init()
    all_needed_hists = [y for x in loaded for y in x.getNeededHistograms()]
    print("Loading Samples")

    all_fields = set()
    by_fields = defaultdict(list)
    for processor in loaded:
        f = processor.getFileFields()
        all_fields |= f
        by_fields[frozenset(f)].append(processor)
    gathered = gatherFilesByPattern(input_files, all_fields)
    with LocalCluster(dashboard_address="localhost:8789") as cluster, Client(
        cluster
    ) as client:

        for keys, processors in by_fields.items():
            fgroups = assembleFileGroup(gathered, keys)
            # print(fgroups)
            # return
            for files in fgroups.values():
                print(files)
                client.submit(
                    processFileGroup,
                    files,
                    processors,
                    use_samples_as_datasets=use_samples_as_datasets,
                    drops=drops,
                )
        client.close()
        cluster.close()
        print("HERE")
    print("HERE1")
