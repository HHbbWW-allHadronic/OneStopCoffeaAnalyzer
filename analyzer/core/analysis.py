from __future__ import annotations
from attrs import define, field
from analyzer.core.serialization import converter, setupConverter

from analyzer.core.analyzer import Analyzer
from analyzer.core.executors import Executor
from analyzer.configuration import CONFIG
from analyzer.utils.config_loading import loadConfigData
from analyzer.utils.load import loadModuleFromPath
from analyzer.utils.querying import Pattern


from analyzer.core.linting import LintConfig

@define
class DatasetDescription:
    pipelines: list[str]
    dataset: Pattern


@define
class Analysis:
    """
    Complete description of an Analysis
    The only options the user should need to provide at runtime are
    the executor with which to run the analysis, and the path of the output data.
    """

    analyzer: Analyzer
    event_collections: list[DatasetDescription]

    extra_module_paths: list[str] = field(factory=list)
    extra_dataset_paths: list[str] = field(factory=list)
    extra_era_paths: list[str] = field(factory=list)
    extra_executors: dict[str, Executor] = field(factory=dict)
    lint_config: LintConfig = field(factory=LintConfig)

    location_priorities: list[str] | None = None


def loadAnalysis(path, variable_name=None):
    if variable_name is None:
        variable_name = CONFIG.analysis_var
    data = loadConfigData(path, variable_name)

    if isinstance(data, Analysis):
        return data

    for path in data.get("extra_module_paths", []):
        from pathlib import Path

        p = Path(path)
        loadModuleFromPath(p.stem, path)

    setupConverter(converter)
    try:
        analysis = converter.structure(data, Analysis)
    except Exception as e:
        from cattrs.errors import BaseValidationError
        from cattrs.v import transform_error

        if isinstance(e, BaseValidationError):
            errors = transform_error(e)
            error_msg = "\n".join([f"  - {err}" for err in errors])
            raise ValueError(
                f"Failed to load analysis due to configuration validation errors:\n{error_msg}"
            ) from None
        raise
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
