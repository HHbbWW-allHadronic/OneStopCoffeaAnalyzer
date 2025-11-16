from __future__ import annotations
import cProfile
import re
from analyzer.core.results import ResultContainer
from rich.logging import RichHandler
import logging
from attrs import define, field
from rich import print
from analyzer.core.serialization import converter
from analyzer.core.analyzer import Analyzer
from analyzer.core.executors import Executor, ExecutionTask
from analyzer.core.datasets import DatasetRepo
from analyzer.utils.load import loadModuleFromPath
import analyzer.core.analysis_modules
import analyzer.core.event_collection
import analyzer.core.executors.executor
import analyzer.core.results

from yaml import CLoader as Loader
import yaml

# Get a logger instance


@define
class DatasetDescription:
    pipelines: list[str]
    collection: str


@define
class Analysis:
    """
    Complete description of an Analysis
    """

    analyzer: Analyzer
    event_collections: list[DatasetDescription]

    extra_module_paths: list[str] = field(factory=list)
    extra_dataset_paths: list[str] = field(factory=list)
    extra_era_paths: list[str] = field(factory=list)
    extra_executors: dict[str, Executor] = field(factory=dict)


def runAnalysis(analysis):
    # default_module_paths = []
    # for path in default_module_paths:
    #     loadRecursive(default_module_paths)
    # for path in analysis.extra_module_paths:
    #     loadRecursive(path)

    dataset_repo = DatasetRepo()

    # default_era_paths = []
    # for path in default_era_paths:
    #     loadRecursive(default_module_paths)
    # for path in analysis.extra_era_paths:
    #     loadRecursive(path)
    #
    for path in analysis.extra_dataset_paths:
        dataset_repo.addFromDirectory(path)
    ds = dataset_repo["test_dataset"]
    meta, sample = ds.getWithMeta("test_sample")
    fs = sample.source.getFileSet()
    t1 = ExecutionTask(fs, meta, ["Signal312", "Signal313"])
    meta, sample = ds.getWithMeta("test_sample2")
    fs = sample.source.getFileSet()
    t2 = ExecutionTask(fs, meta, ["Signal312", "Signal313"])

    executor = analysis.extra_executors["test"]

    all_results = None
    for result in executor.run(analysis.analyzer, [t1,t2]):
        if all_results is None:
            all_results=result
        else:
            all_results += result

    with open("test.result", 'wb') as f:
        f.write(all_results.toBytes())

def setupConverter(conv):
    analyzer.core.analysis_modules.configureConverter(converter)
    analyzer.core.event_collection.configureConverter(converter)
    analyzer.core.executors.executor.configureConverter(converter)
    analyzer.core.results.configureConverter(converter)


def main():
    with open("test.yaml") as f:
        data = yaml.load(f, Loader=Loader)

    import analyzer.modules

    for path in data.get("extra_module_paths", []):
        loadModuleFromPath(path)

    setupConverter(converter)

    a = converter.structure(data, Analysis)

    # profiler = cProfile.Profile()
    # profiler.enable()
    runAnalysis(a)
    # profiler.disable()
    # filename = 'prof.prof'  # You can change this if needed
    # profiler.dump_stats(filename)


if __name__ == "__main__":
    main()
