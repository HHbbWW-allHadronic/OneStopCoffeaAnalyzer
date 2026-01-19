from pathlib import Path

from analyzer.core.analysis import loadAnalysis, getSamples
from collections import defaultdict
from analyzer.configuration import CONFIG
from analyzer.core.executors import getPremadeExcutors, ExecutionTask
from analyzer.utils.structure_tools import getWithMeta, globWithMeta
from analyzer.core.results import loadResults
from analyzer.core.event_collection import buildMissingFileset
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


def getMatchedCollections(dataset_repo, descs):
    ret = defaultdict(list)
    for k in dataset_repo:
        for desc in descs:
            matched = desc.dataset.match(k)
            if matched:
                ret[k].append(desc)
    return ret


def getTasks(dataset_repo, era_repo, dataset_descs, location_priorities=None):
    todo = []
    matched = getMatchedCollections(dataset_repo, dataset_descs)
    if any(len(x) != 1 for x in matched.values()):
        raise RuntimeError("More than one matching pattern.")
    todo = [(k, x[0].pipelines) for k, x in matched.items()]
    ret = []
    for dataset_name, pipelines in todo:
        dataset = dataset_repo[dataset_name]
        for sample in dataset:
            sample, meta = getWithMeta(dataset, sample.name)
            meta = dict(meta)
            meta["era"] = era_repo[meta["era"]]
            file_set = sample.source.getFileSet(location_priorities=location_priorities)
            ret.append(
                ExecutionTask(
                    file_set=file_set,
                    metadata=meta,
                    pipelines=pipelines,
                    output_name=meta["dataset_name"] + "__" + meta["sample_name"],
                )
            )
    return ret


def getTasksExplicit(
    dataset_repo, era_repo, dataset_descs, samples, location_priorities=None
):
    ret = []
    for dataset_name, sample_name in samples:
        matched = [x for x in dataset_descs if x.dataset.match(dataset_name)]
        if len(matched) != 1:
            raise RuntimeError("More than one matching pattern")
        pipelines = matched[0].pipelines
        dataset = dataset_repo[dataset_name]
        sample, meta = getWithMeta(dataset, sample_name)
        meta = dict(meta)
        meta["era"] = era_repo[meta["era"]]
        file_set = sample.source.getFileSet(location_priorities=location_priorities)
        ret.append(
            ExecutionTask(
                file_set=file_set,
                metadata=meta,
                pipelines=pipelines,
                output_name=meta["dataset_name"] + "__" + meta["sample_name"],
            )
        )
    return ret


def runTasks(analyzer, tasks, executor, output, max_sample_events=None):
    needed_resources = set()
    for task in tasks:
        needed_resources |= set(analyzer.neededResources(task.metadata))
    needed_resources = list(needed_resources)
    executor.setup(needed_resources)

    saver = Saver(output)

    for result in executor.run(analyzer, tasks, max_sample_events=max_sample_events):
        saver(result.output_name, result.result)


def runFromPath(
    path,
    output,
    executor_name,
    max_sample_events=None,
    filter_samples=None,
    limit_pipelines=None,
    return_analyzer=False,
):
    logger.info(f'Running analysis from path "{path}" with executor {executor_name}')
    output = Path(output)
    analysis = loadAnalysis(path)
    dataset_repo, era_repo = getRepos(
        analysis.extra_dataset_paths, analysis.extra_era_paths
    )
    all_executors = getPremadeExcutors()
    all_executors = {**all_executors, **analysis.extra_executors}

    executor = all_executors[executor_name]

    tasks = getTasks(
        dataset_repo,
        era_repo,
        analysis.event_collections,
        location_priorities=analysis.location_priorities,
    )
    logger.info("Initializing analyzer")
    for t in tasks:
        analysis.analyzer.initModules(t.metadata)
    logger.info(
        f"Preparing to run {len(tasks)} tasks. Max events per sample is {max_sample_events}"
    )
    runTasks(
        analysis.analyzer, tasks, executor, output, max_sample_events=max_sample_events
    )
    if return_analyzer:
        return analysis.analyzer


def patchFromPath(
    path, existing, output, executor_name, filter_samples=None, limit_pipelines=None
):
    logger.info(f'Running analysis from path "{path}" with executor {executor_name}')
    output = Path(output)
    analysis = loadAnalysis(path)
    dataset_repo, era_repo = getRepos(
        analysis.extra_dataset_paths, analysis.extra_era_paths
    )
    all_executors = getPremadeExcutors()
    all_executors.update(analysis.extra_executors)
    executor = all_executors[executor_name]

    results = loadResults(existing, peek_only=True)

    provenances = globWithMeta(results, ("*", "*", "_provenance"))
    all_samples = getSamples(analysis, dataset_repo)
    present = set((x["dataset_name"], x["sample_name"]) for _, x in provenances)
    missing = sorted(all_samples - present)
    missing_tasks = getTasksExplicit(
        dataset_repo, era_repo, analysis.event_collections, missing
    )

    # matched = getMatchedCollections(dataset_repo, analysis.event_collections)
    all_tasks = missing_tasks

    for provenance, meta in provenances:
        dataset_name = meta["dataset_name"]
        sample_name = meta["sample_name"]
        n_events = meta["n_events"]
        if provenance.chunked_events == n_events:
            continue
        s = dataset_repo[dataset_name][sample_name]
        source = s.source
        task = getTasksExplicit(
            dataset_repo,
            era_repo,
            analysis.event_collections,
            [(dataset_name, sample_name)],
        )[0]
        task.file_set = buildMissingFileset(source, provenance.file_set)
        if not task.file_set.empty:
            all_tasks.append(task)

    logger.info(f"Preparing to run {len(all_tasks)} tasks.")
    runTasks(analysis.analyzer, all_tasks, executor, output)
