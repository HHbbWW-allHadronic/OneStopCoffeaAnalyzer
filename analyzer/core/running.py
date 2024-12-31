import itertools as it
import logging
import sys
import pickle as pkl
from dataclasses import dataclass, field
from typing import Any, Union
from pathlib import Path


from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId
from coffea.nanoevents import NanoAODSchema

from analyzer.core.region_analyzer import RegionAnalyzer, getParamsSample
from analyzer.core.configuration import getSubSectors, loadDescription
from analyzer.core.analyzer import Analyzer
from analyzer.core.executor import AnalysisTask

if CONFIG.PRETTY_MODE:
    from rich import print


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


def runFromPath(path, output, executor_name, save_separate=False):
    import analyzer.modules

    output=Path(output)
    description = loadDescription(path)
    if executor_name not in description.executors:
        raise KeyError()
        
    executor = description.executors[executor_name]
    
    dataset_repo = DatasetRepo.getConfig()
    era_repo = EraRepo.getConfig()
    subsectors = getSubSectors(description, dataset_repo, era_repo)
    tasks = makeTasks(
        subsectors, dataset_repo, era_repo, description.file_config.model_dump()
    )

    tasks = {task.sample_id: task for task in tasks}
    results = executor.run(tasks)

    if save_separate:
        output.mkdir(exist_ok=True, parents=True)
        for k,v in results.items():
            with open(output/ f"{k}.pkl", "wb") as f:
                pkl.dump({k: v.model_dump()}, f)
    else:
        with open(output, "wb") as f:
            pkl.dump({x: y.model_dump() for x, y in results.items()}, f)


