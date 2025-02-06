import concurrent.futures as cf
import itertools as it

import matplotlib as mpl
from analyzer.core.results import loadSampleResultFromPaths, makeDatasetResults
from analyzer.utils.progress import progbar
from rich import print
from rich.progress import Progress

from .plots.mplstyles import loadStyles
from .processors import PostProcessorType, postprocess_catalog
from .registry import loadPostprocessors


def initProcess():
    mpl.use("Agg")
    loadStyles()


def run(tasks, parallel):
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Processing...", total=len(tasks))
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


def runPostprocessors(config, input_files, parallel=8):
    loadStyles()
    print("Loading Postprocessors")
    loaded, catalog, drops, use_samples_as_datasets = loadPostprocessors(config)
    all_needed_hists = [y for x in loaded for y in x.getNeededHistograms()]
    print("Loading Samples")
    sample_results = loadSampleResultFromPaths(input_files, include=all_needed_hists)

    def dropSampleFunction(sid):
        if not drops:
            return False
        return any(pattern.match(sid.sample_name) for pattern in drops)

    dataset_results = makeDatasetResults(
        sample_results,
        drop_sample_fn=dropSampleFunction,
        include_samples_as_datasets=use_samples_as_datasets,
    )
    print("Ready to Process ")
    sector_results = list(
        it.chain.from_iterable(r.sector_results for r in dataset_results.values())
    )

    tasks, items = [], []
    acc_tasks = []
    for processor in progbar(loaded):
        processor.init()
        if processor.postprocessor_type == PostProcessorType.Normal:
            t, i = processor.getExe(sector_results)
            tasks += t
            items += i
        elif processor.postprocessor_type == PostProcessorType.Accumulator:
            t = processor.getExe(sector_results)
            acc_tasks += t
    if tasks:
        run(tasks, parallel)

    if tasks and catalog:
        with open(catalog, "wb") as f:
            f.write(postprocess_catalog.dump_json(items, indent=2))

    if acc_tasks:
        run(acc_tasks, parallel)
