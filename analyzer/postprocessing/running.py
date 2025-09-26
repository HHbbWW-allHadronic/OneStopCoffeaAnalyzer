import concurrent.futures as cf
import itertools as it

import matplotlib as mpl
from analyzer.utils.debugging import jumpIn
from collections import defaultdict
from analyzer.core.results import (
    loadSampleResultFromPaths,
    makeDatasetResults,
    gatherFilesByPattern,
    combineResults,
    openAndLoad,
    makeFilePeekMap,
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
    Queue,
    as_completed,
    WorkerPlugin,
)

from .plots.mplstyles import loadStyles

# from .processors import PostProcessorType, postprocess_catalog
from .registry import loadPostprocessors


import analyzer.postprocessing.processors  # noqa
import analyzer.postprocessing.exporting  # noqa
import analyzer.postprocessing.basic_histograms  # noqa
import analyzer.postprocessing.cutflows  # noqa
from .plots.mplstyles import loadStyles


def initProcess():
    mpl.use("Agg")
    loadStyles()


class LoadStyles(WorkerPlugin):
    def setup(self, worker):
        loadStyles()

    def teardown(self, worker):
        pass


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


def processFileGroup(
    queue, files, postprocessors, use_samples_as_datasets=False, drops=None
):
    all_needed_hists = [y for x in postprocessors for y in x.getNeededHistograms()]
    # sample_results = combineResults(files)
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
    if queue is not None:
        client = get_client()
        for x in it.chain.from_iterable(
            (processor.getExe(sector_results) for processor in postprocessors)
        ):
            queue.put(client.submit(x))
    else:
        for x in it.chain.from_iterable(
            (processor.getExe(sector_results) for processor in postprocessors)
        ):
            x()


def assembleFileGroup(param_dict, fields):
    ret = defaultdict(set)
    for k, v in param_dict.items():
        newk = frozenset((x, y) for x, y in k if x in fields)
        ret[newk] |= v
    return ret


__file_cache = {}


def getFile(client, path):
    if str(path) in __file_cache:
        return __file_cache[str(path)]

    ret = client.submit(openAndLoad, path)
    __file_cache[str(path)] = ret
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
    mp = makeFilePeekMap(input_files)
    to_run = defaultdict(list)
    for processor in loaded:
        p = processor.neededFileSets(mp)
        for s in p:
            to_run[s].append(processor)
    # print(to_run)

    queue = Queue()
    futures = []
    if parallel:
        with LocalCluster(
            dashboard_address="localhost:8789",
            preload=["analyzer.postprocessing.style"],
        ) as cluster, Client(cluster) as client:
            client.register_plugin(LoadStyles())
            for fset, processors in to_run.items():
                futures.append(
                    client.submit(
                        processFileGroup,
                        queue,
                        fset,
                        processors,
                        use_samples_as_datasets=use_samples_as_datasets,
                        drops=drops,
                    )
                )
            completed = as_completed(futures)
            for f in track(completed):
                while queue.qsize() > 0:
                    gotten = queue.get()
                    completed.add(gotten)
                f.result()
                f.cancel()
    else:
        for fset, processors in to_run.items():
            processFileGroup(
                None,
                fset,
                processors,
                use_samples_as_datasets=use_samples_as_datasets,
                drops=drops,
            )
