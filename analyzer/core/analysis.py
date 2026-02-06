from __future__ import annotations
from attrs import define, field
from analyzer.core.serialization import converter, setupConverter

from analyzer.core.analyzer import Analyzer
from analyzer.core.executors import Executor
from analyzer.utils.load import loadModuleFromPath
from analyzer.utils.querying import Pattern

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
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

    location_priorities: list[str] | None = None


def loadAnalysis(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=Loader)

    for path in data.get("extra_module_paths", []):
        from pathlib import Path

        p = Path(path)
        loadModuleFromPath(p.stem, path)

    setupConverter(converter)
    analysis = converter.structure(data, Analysis)
    return analysis


def getSamples(analysis, dataset_repo, filter_dataset=None, filter_sample=None):
    todo = set()
    for desc in analysis.event_collections:
        ds = set(x for x in dataset_repo if desc.dataset.match(x))
        todo |= ds

    ret = set()
    for dataset_name in todo:
        if filter_dataset is not None and not filter_dataset.match(dataset_name):
            continue
        dataset = dataset_repo[dataset_name]
        for sample in dataset:
            if filter_sample is not None and not filter_sample.match(sample.name):
                continue
            ret.add((dataset_name, sample.name))
    return ret
