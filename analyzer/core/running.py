import contextlib
import pickle as pkl
from pathlib import Path

from analyzer.core.analysis_modules import MODULE_REPO
from analyzer.core.analyzer import Analyzer
from analyzer.core.configuration import getSubSectors, loadDescription
from analyzer.core.executor import AnalysisTask
from analyzer.core.patching import getSamplePatch
from analyzer.core.results import loadSampleResultFromPaths
from analyzer.datasets import DatasetRepo, EraRepo
from rich import print
import logging


logger = logging.getLogger(__name__)


def makeTasks(subsectors, dataset_repo, era_repo, file_retrieval_kwargs):
    ret = []
    for sample_id, region_analyzers in subsectors.items():
        params = dataset_repo[sample_id].params
        params.dataset.populateEra(era_repo)
        u = AnalysisTask(
            sample_id=sample_id,
            sample_params=params,
            file_set=dataset_repo[sample_id].getFileSet(file_retrieval_kwargs),
            analyzer=Analyzer(region_analyzers),
        )
        ret.append(u)
    return ret


@contextlib.contextmanager
def openNoOverwrite(file_name, *args, **kwargs):
    p = Path(file_name)
    orig_stem = p.stem
    i = 1
    while p.exists():
        p = p.with_stem(orig_stem + "_" + str(i))
        i += 1

    handle = open(p, *args, **kwargs)
    yield handle
    handle.close()


def saveResults(results, output, save_separate=False):
    if not results:
        return
    output = Path(output)
    if save_separate:
        output.mkdir(exist_ok=True, parents=True)
        for k, v in results.items():
            with openNoOverwrite(output / f"{k}.pkl", "wb") as f:
                pkl.dump({k: v.model_dump()}, f)
    else:
        output.parent.mkdir(exist_ok=True, parents=True)
        with openNoOverwrite(output, "wb") as f:
            pkl.dump({x: y.model_dump() for x, y in results.items()}, f)


def makeSaveCallback(output):
    def inner(key, result):
        output.mkdir(exist_ok=True, parents=True)
        output_file = output / f"{key}.pkl"
        logger.info(f"Saving key {key} to \"{output_file}\"")
        with openNoOverwrite(output_file, "wb") as f:
            pkl.dump({key: result.model_dump()}, f)
    return inner


def runFromPath(path, output, executor_name, save_separate=False, test_mode=False):
    import analyzer.modules

    output = Path(output)
    description = loadDescription(path)

    dataset_repo = DatasetRepo.getConfig()
    era_repo = EraRepo.getConfig()
    subsectors = getSubSectors(description, dataset_repo, era_repo)

    for k, v in subsectors.items():
        print(f"{k} => {[x.region_name for x in v]}")

    tasks = makeTasks(
        subsectors, dataset_repo, era_repo, description.file_config.model_dump()
    )

    tasks = {task.sample_id: task for task in tasks}

    if executor_name not in description.executors:
        raise KeyError(f"Unknown executor {executor_name}")

    executor = description.executors[executor_name]
    executor.test_mode = test_mode
    executor.setup()
    if hasattr(executor, "output_dir") and executor.output_dir is None:
        executor.output_dir = str(output)

    if save_separate:
        callback = makeSaveCallback(output)
        results = executor.run(tasks, result_complete_callback=callback)
    else:
        results = executor.run(tasks)
        saveResults(results, output)


def runPackagedTask(packaged_task, output=None, output_dir=None, save_separate=False):
    import analyzer.modules

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    iden = packaged_task.identifier
    output = output or iden

    task = packaged_task.task
    executor = packaged_task.executor

    task.analyzer.ensureFunction(MODULE_REPO)

    results = executor.run({task.sample_id: task})
    saveResults(results, output_dir / output, save_separate=save_separate)


def mergeResults(paths, output_path, save_separate=False):
    result = loadSampleResultFromPaths(paths)
    saveResults(results, output, save_separate=save_separate)


def patchFromPath(
    paths,
    output,
    executor_name,
    description_path,
    save_separate=False,
    ignore_ret_prefs=False,
):
    import analyzer.modules

    output = Path(output)
    inputs = [Path(path) for path in paths]
    if output in inputs:
        raise RuntimeError()
    description = loadDescription(description_path)
    executor = description.executors[executor_name]
    executor.setup()
    if hasattr(executor, "output_dir") and executor.output_dir is None:
        executor.output_dir = str(output)

    dataset_repo = DatasetRepo.getConfig()
    sample_results = loadSampleResultFromPaths(inputs)
    patches = [getSamplePatch(s, dataset_repo) for s in sample_results.values()]
    patches = [p for p in patches if not p.file_set.empty]
    for p in patches:
        p.analyzer.ensureFunction(MODULE_REPO)
        if ignore_ret_prefs:
            p.file_set.file_retrieval_kwargs = {}
    tasks = {task.sample_id: task for task in patches}

    if save_separate:
        callback = makeSaveCallback(output)
        results = executor.run(tasks, result_complete_callback=callback)
    else:
        results = executor.run(tasks)
        saveResults(results, output)
