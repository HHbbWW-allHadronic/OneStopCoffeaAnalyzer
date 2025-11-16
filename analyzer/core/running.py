import logging
from pathlib import Path

from analyzer.core.analysis_modules import MODULE_REPO
from analyzer.core.analyzer import Analyzer
from analyzer.core.configuration import loadDescription, iterSubsectors
from analyzer.core.executors import AnalysisTask
from analyzer.core.patching import getSamplePatch
from analyzer.core.results import loadSampleResultFromPaths
from analyzer.datasets import DatasetRepo, EraRepo
from rich import print

logger = logging.getLogger(__name__)


def getUniqueFilename(file_name):
    p = Path(file_name)
    orig_stem = p.stem
    i = 1
    while p.exists():
        p = p.with_stem(orig_stem + "_" + str(i))
        i += 1
    return p


class Saver:
    def __init__(self, output):
        self.output = output

    def __call__(self, key, result):
        self.output.mkdir(exist_ok=True, parents=True)
        output_file = self.output / f"{key}.result"
        real_path = getUniqueFilename(output_file)
        logger.info(f"Saving file '{real_path}'")
        print(f"Saving file '{real_path}'")
        with open(real_path, "wb") as f:
            f.write(result)


def runFromPath(path, output, executor_name, filter_samples=None, limit_pipelines=None):
    output = Path(output)
    description = loadDescription(path)

    dataset_repo = DatasetRepo.getConfig()
    era_repo = EraRepo.getConfig()

    if executor_name not in description.executors:
        raise KeyError(f"Unknown executor {executor_name}")

    executor = description.executors[executor_name]
    executor.test_mode = test_mode
    with executor:
        if hasattr(executor, "output_dir") and executor.output_dir is None:
            executor.output_dir = str(output)
        callback = Saver(output)
        results = executor.run(tasks)


# def runPackagedTask(packaged_task, output=None, output_dir=None):
#     import analyzer.modules  # noqa
#
#     if output_dir is None:
#         output_dir = Path(".")
#     else:
#         output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True, parents=True)
#
#     iden = packaged_task.identifier
#     output = output or iden
#
#     task = packaged_task.task
#     executor = packaged_task.executor
#
#     task.analyzer.ensureFunction(MODULE_REPO)
#
#     results = executor.run({task.sample_id: task})
#     saveResults(results, output_dir / output, save_separate=True)
#
#
# def patchFromPath(
#     paths,
#     output,
#     executor_name,
#     description_path,
#     ignore_ret_prefs=False,
#     threshhold=0.95,
# ):
#     import analyzer.modules  # noqa
#
#     output = Path(output)
#     inputs = [Path(path) for path in paths]
#     if output in inputs:
#         raise RuntimeError()
#     description = loadDescription(description_path)
#     executor = description.executors[executor_name]
#     if hasattr(executor, "output_dir") and executor.output_dir is None:
#         executor.output_dir = str(output)
#
#     dataset_repo = DatasetRepo.getConfig()
#     era_repo = EraRepo.getConfig()
#
#     raw_loaded = loadSampleResultFromPaths(
#         inputs, include=[], show_progress=True, parallel=None, peek_only=True
#     )
#     loaded = list(raw_loaded.values())
#
#     to_load_real = set()
#
#     for peek in loaded:
#         peek.params.sample_id
#         exp = peek.params.n_events
#         val = peek.processed_events
#         frac_done = val / exp
#         if frac_done < threshhold:
#             to_load_real |= peek._from_files
#
#     sample_results = loadSampleResultFromPaths(
#         to_load_real, include=[], show_progress=True
#     )
#     patches = [getSamplePatch(s, dataset_repo) for s in sample_results.values()]
#     patches = [p for p in patches if not p.file_set.empty]
#
#     for p in patches:
#         p.analyzer.ensureFunction(MODULE_REPO)
#         if ignore_ret_prefs:
#             p.file_set.file_retrieval_kwargs = {
#                 "location_priority_regex": [".*(T0|T1|T2).*", ".*"]
#             }
#
#     #subsectors = getSubSectors(description, dataset_repo, era_repo)
#     subsectors = list(iterSubsectors(description, dataset_repo, era_repo))
#     #unknown_sample_tasks = makeTasks(
#     #    {x: y for x, y in subsectors.items() if x not in raw_loaded},
#     #    dataset_repo,
#     #    era_repo,
#     #    description.file_config.model_dump(),
#     #)
#     unknown_sample_tasks = list(iterTasks(
#         [x for x in subsectors if x[0] not in raw_loaded], dataset_repo, era_repo, description.file_config.model_dump()
#     ))
#
#     final_tasks = unknown_sample_tasks + patches
#
#     callback = Saver(output)
#     with executor:
#         results = executor.run(final_tasks, result_complete_callback=callback)
#
#
# def describeFromPath(path, output, executor_name, test_mode=False, filter_samples=None):
#     import analyzer.modules  # noqa
#
#     output = Path(output)
#     description = loadDescription(path)
#
#     dataset_repo = DatasetRepo.getConfig()
#     era_repo = EraRepo.getConfig()
#     subsectors = iterSubsectors(
#         description, dataset_repo, era_repo, filter_samples=filter_samples
#     )
#     print(list(subsectors))
#     print(f"Saving to {output}")
#     print(f"Running using executor {executor_name}")
