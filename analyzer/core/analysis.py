from __future__ import annotations


from rich.logging import RichHandler
import logging

# Define the log message format
FORMAT = "%(message)s"

# Configure basic logging with RichHandler
logging.basicConfig(
    level="WARNING",  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    format=FORMAT,
    handlers=[RichHandler()],
)

from attrs import define, field
from rich import print
from analyzer.core.serialization import converter
from analyzer.core.analyzer import Analyzer
from analyzer.core.executors import Executor , ExecutionTask
from analyzer.core.datasets import DatasetRepo
from analyzer.utils.load import loadModuleFromPath
import analyzer.core.analysis_modules
import analyzer.core.event_collection
import analyzer.core.executors.executor

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
    ds=dataset_repo["test_dataset"]
    meta,sample = ds.getWithMeta("test_sample")
    executor = analysis.extra_executors["test"]
    fs = sample.source.getFileSet()

    t = ExecutionTask(fs, meta, ["Signal312", "Signal313"])
    for result in executor.run(analysis.analyzer, [t]):
        print(result)



def setupConverter(conv):
    analyzer.core.analysis_modules.configureConverter(converter)
    analyzer.core.event_collection.configureConverter(converter)
    analyzer.core.executors.executor.configureConverter(converter)



def main():
    with open("test.yaml") as f:
        data = yaml.load(f, Loader=Loader)

    import analyzer.modules

    for path in data.get("extra_module_paths",[]):
        loadModuleFromPath(path)

    setupConverter(converter)

    a = converter.structure(data, Analysis)
    runAnalysis(a)


if __name__ == "__main__":
    main()
    
