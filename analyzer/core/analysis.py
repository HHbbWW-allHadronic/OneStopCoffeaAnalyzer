from __future__ import annotations
import cProfile
import re
from analyzer.core.results import ResultContainer
from rich.logging import RichHandler
from analyzer.utils.structure_tools import getWithMeta
import logging
from attrs import define, field
from rich import print
from analyzer.core.serialization import converter, setupConverter
from analyzer.core.analyzer import Analyzer
from analyzer.core.executors import Executor, ExecutionTask
from analyzer.core.datasets import DatasetRepo
from analyzer.utils.load import loadModuleFromPath
from analyzer.utils.querying import Pattern

from yaml import CLoader as Loader
import yaml

# Get a logger instance


@define
class DatasetDescription:
    pipelines: list[str]
    dataset: Pattern


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

def loadAnalysis(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=Loader)
    import analyzer.modules
    for path in data.get("extra_module_paths", []):
        loadModuleFromPath(path)

    setupConverter(converter)
    analysis = converter.structure(data, Analysis)
    return analysis


def main():
    a = loadAnalysis("test.yaml")
    runAnalysis(a)
    
if __name__ == "__main__":
    main()


