import pickle as pkl
from pathlib import Path


from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo

from analyzer.core.configuration import getSubSectors, loadDescription
from analyzer.core.analyzer import Analyzer
from analyzer.core.executor import AnalysisTask
from .analysis_modules import MODULE_REPO


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


def saveResults(results, output, save_separate=False):
    if not results:
        return
    if save_separate:
        output.mkdir(exist_ok=True, parents=True)
        for k, v in results.items():
            with open(output / f"{k}.pkl", "wb") as f:
                pkl.dump({k: v.model_dump()}, f)
    else:
        with open(output, "wb") as f:
            pkl.dump({x: y.model_dump() for x, y in results.items()}, f)


def runFromPath(path, output, executor_name, save_separate=False):
    output = Path(output)
    description = loadDescription(path)
    if executor_name not in description.executors:
        raise KeyError()

    executor = description.executors[executor_name]
    executor.setup()
    if hasattr(executor, "output_dir") and executor.output_dir is None:
        executor.output_dir = str(output)
        

    dataset_repo = DatasetRepo.getConfig()
    era_repo = EraRepo.getConfig()
    subsectors = getSubSectors(description, dataset_repo, era_repo)
    tasks = makeTasks(
        subsectors, dataset_repo, era_repo, description.file_config.model_dump()
    )

    tasks = {task.sample_id: task for task in tasks}
    results = executor.run(tasks)

    saveResults(results, output, save_separate=save_separate)


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
