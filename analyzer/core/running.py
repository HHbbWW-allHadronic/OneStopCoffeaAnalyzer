import logging
from pathlib import Path

from rich import print
from analyzer.core.analysis import loadAnalysis
from analyzer.configuration import CONFIG
from analyzer.core.executors import getPremadeExcutors, ExecutionTask
from analyzer.utils.structure_tools import getWithMeta
from analyzer.logging import logger


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
        with open(real_path, "wb") as f:
            f.write(result)


def getRepos(extra_dataset_paths=None, extra_era_paths=None):
    from analyzer.core.datasets import DatasetRepo
    from analyzer.core.era import EraRepo

    extra_dataset_paths = extra_dataset_paths or []
    extra_era_paths = extra_era_paths or []

    default_dataset_paths = CONFIG.datasets.default_dataset_paths
    default_era_paths = CONFIG.datasets.default_era_paths

    dataset_repo = DatasetRepo()
    era_repo = EraRepo()

    for path in default_dataset_paths + extra_dataset_paths:
        dataset_repo.addFromDirectory(path)
    for path in default_era_paths + extra_era_paths:
        era_repo.addFromDirectory(path)

    return dataset_repo, era_repo


def getTasks(dataset_repo, era_repo, dataset_descs):
    todo = []
    for desc in dataset_descs:
        ds = [(x, desc.pipelines) for x in dataset_repo if desc.dataset.match(x)]
        todo.extend(ds)
    ret = []
    for dataset_name, pipelines in todo:
        dataset = dataset_repo[dataset_name]
        for sample in dataset:
            sample, meta = getWithMeta(dataset, sample.name)
            meta = dict(meta)
            meta["era"] = era_repo[meta["era"]]
            file_set = sample.source.getFileSet()
            ret.append(
                ExecutionTask(
                    file_set=file_set,
                    metadata=meta,
                    pipelines=pipelines,
                    output_name=meta["dataset_name"] + "__" + meta["sample_name"],
                )
            )
    return ret


def getPatches(dataset_repo, era_repo, dataset_descs, existing_file_sets):
    todo = []
    for desc in dataset_descs:
        ds = [(x, desc.pipelines) for x in dataset_repo if desc.dataset.match(x)]
        todo.extend(ds)
    ret = []
    for dataset_name, pipelines in todo:
        dataset = dataset_repo[dataset_name]
        for sample in dataset:
            sample, meta = getWithMeta(dataset, sample.name)
            meta["era"] = era_repo[meta["era"]]
            file_set = sample.source.getFileSet()
            ret.append(
                ExecutionTask(
                    file_set=file_set,
                    metadata=meta,
                    pipelines=pipelines,
                    output_name=meta["dataset_name"] + "__" + meta["sample_name"],
                )
            )
    return ret


def runFromPath(path, output, executor_name, filter_samples=None, limit_pipelines=None):

    from analyzer.core.datasets import DatasetRepo
    from analyzer.core.era import EraRepo

    logger.info(f'Running analysis from path "{path}" with executor {executor_name}')
    output = Path(output)
    analysis = loadAnalysis(path)
    dataset_repo, era_repo = getRepos(
        analysis.extra_dataset_paths, analysis.extra_era_paths
    )
    all_executors = getPremadeExcutors()
    all_executors.update(analysis.extra_executors)

    executor = all_executors[executor_name]

    tasks = getTasks(dataset_repo, era_repo, analysis.event_collections)
    logger.info(f"Preparing to run {len(tasks)} tasks.")

    needed_resources = set()
    for task in tasks:
        needed_resources |= set(analysis.analyzer.neededResources(task.metadata))
    needed_resources = list(needed_resources)
    executor.setup(needed_resources)

    saver = Saver(output)

    for result in executor.run(analysis.analyzer, tasks):
        saver(result.output_name, result.result)


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


def main():
    runFromPath("test.yaml", "TESTRESULTS", "test")


if __name__ == "__main__":
    main()
