import concurrent.futures as cf
import itertools as it

import matplotlib as mpl
from analyzer.core.results import loadSampleResultFromPaths, makeDatasetResults
from rich import print
from rich.progress import Progress

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


def runPostprocessors(config, input_files, parallel=8):
    loadStyles()
    print("Loading Postprocessors")
    loaded, catalog, drops, use_samples_as_datasets, num_files = loadPostprocessors(
        config
    )
    all_needed_hists = [y for x in loaded for y in x.getNeededHistograms()]
    print("Loading Samples")

    if num_files:
        to_run = chunks(input_files, num_files)
    else:
        to_run = [input_files]

    for files in to_run:
        sample_results = loadSampleResultFromPaths(files, include=all_needed_hists)

        def dropSampleFunction(sid):
            if drops is None:
                return False
            return drops.match(sid.sample_name)

        dataset_results = makeDatasetResults(
            sample_results,
            drop_sample_fn=dropSampleFunction,
            include_samples_as_datasets=use_samples_as_datasets,
        )
        print("Ready to Process ")
        sector_results = list(
            it.chain.from_iterable(r.sector_results for r in dataset_results.values())
        )
        for processor in loaded:
            processor.init()
        run(
            it.chain.from_iterable(
                (processor.getExe(sector_results) for processor in loaded)
            ),
            parallel,
        )
