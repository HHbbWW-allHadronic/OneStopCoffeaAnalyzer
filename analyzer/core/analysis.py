from __future__ import annotations
from attrs import define, field
from analyzer.core.serialization import converter
from analyzer.core.analyzer import Analyzer
from analyzer.core.executors import Executor
from analyzer.core.datasets import DatasetRepo
from analyzer.utils.load import loadModuleFromPath
import analyzer.core.analysis_modules
import analyzer.core.event_collection
from yaml import CLoader as Loader
import yaml



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
    # extra_executors: dict[str, Executor] = field(factory=dict)





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
    breakpoint()
    # for path in analysis.extra_dataset_paths:
    #     loadRecursive(path)



def setupConverter(conv):
    analyzer.core.analysis_modules.configureConverter(converter)
    analyzer.core.event_collection.configureConverter(converter)



def main():
    with open("test.yaml") as f:
        data = yaml.load(f, Loader=Loader)

    import analyzer.modules

    for path in data.get("extra_module_paths",[]):
        loadModuleFromPath(path)

    setupConverter(converter)

    a = converter.structure(data, Analysis)
    runAnalysis(a)
    print(a)


if __name__ == "__main__":
    main()
    
