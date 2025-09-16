import enum
from collections import defaultdict
import re
from typing import Any, ClassVar

import yaml

from pydantic import BaseModel, Field

from .analysis_modules import MODULE_REPO

import logging

import analyzer.core.region_analyzer as ra
import analyzer.core.executors as executors
from analyzer.utils.querying import PatternExpression, Pattern
from analyzer.utils.debugging import jumpIn

from collections import defaultdict

logger = logging.getLogger(__name__)


class AnalysisStage(str, enum.Enum):
    Preselection = "Preselection"
    Correction = "Correction"
    ObjectDefinition = "ObjectDefinition"
    Selection = "Selection"
    Weights = "Weights"
    Categorization = "Categorization"
    Histogramming = "Histogramming"


class ModuleDescription(BaseModel):
    name: str
    constraint: PatternExpression | None = None
    config: list[dict[str, Any]] | dict[str, Any] | None = None


class RegionDescription(BaseModel):
    name: str
    use_region: bool = True
    forbid_data: bool = False
    description: str = ""

    selection: list[ModuleDescription] = Field(default_factory=list)
    objects: list[ModuleDescription] = Field(default_factory=list)
    corrections: list[ModuleDescription] = Field(default_factory=list)
    preselection: list[ModuleDescription] = Field(default_factory=list)
    preselection_histograms: list[ModuleDescription] = Field(default_factory=list)
    categories: list[ModuleDescription] = Field(default_factory=list)
    histograms: list[ModuleDescription] = Field(default_factory=list)
    weights: list[ModuleDescription] = Field(default_factory=list)


class ExecutionConfig(BaseModel):
    cluster_type: str = "local"
    max_workers: int = 20
    step_size: int = 100000
    worker_memory: str | None = "4GB"
    dashboard_address: str | None = None
    schedd_address: str | None = None
    worker_timeout: int = 3600
    extra_files: list[str] | None = None


class FileConfig(BaseModel):
    location_priority_regex: list[str] = [
        ".*FNAL.*",
        ".*US.*",
        ".*(DE|IT|CH|FR).*",
        ".*(T0|T1|T2).*",
        "eos",
    ]
    use_replicas: bool = True


class DatasetRegionElement(BaseModel):
    dataset: Pattern
    regions: list[str]


class AnalysisDescription(BaseModel):
    name: str
    executors: dict[str, executors.AnyExecutor]
    # execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    file_config: FileConfig = Field(default_factory=FileConfig)
    datasets: dict[str, list[str] | str] | list[DatasetRegionElement]
    regions: list[RegionDescription]
    general_config: dict[str, Any] = Field(default_factory=dict)
    special_region_names: ClassVar[tuple[str]] = ("All",)

    def getRegion(self, name):
        try:
            return next(x for x in self.regions if x.name == name)
        except StopIteration:
            raise KeyError(f'No region "{name}"')

    def __eq__(self, other):
        def asTuple(ad):
            return (ad.name, ad.samples, ad.regions)

        return asTuple(self) == asTuple(other)


def loadDescription(input_path):
    with open(input_path, "rb") as config_file:
        data = yaml.safe_load(config_file)
    return AnalysisDescription(**data)


def iterSubsectors(description, dataset_repo, era_repo, filter_samples=None):
    by_dataset = defaultdict(list)
    for e in description.datasets:
        if isinstance(e, DatasetRegionElement):
            todo = [(x, e.regions) for x in dataset_repo if e.dataset.match(x)]
        else:
            dataset_name, regions = e, description.datasets[e]
            todo = [(dataset_name, regions)]
        for dataset_name, regions in todo:
            if isinstance(regions, str) and regions == "All":
                regions = [r.name for r in description.regions]
            for r in regions:
                by_dataset[dataset_name].append(r)
    
    for dataset_name, regions in by_dataset.items():
        # logger.debug(
        #     f'Getting analyzer for dataset "{dataset_name}" and regions "{regions}"'
        # )
        dataset = dataset_repo[dataset_name]
        for sample in dataset.samples:
            if filter_samples is not None and not any(
                f.match(sample.name) for f in filter_samples
            ):
                continue

            logger.info(
                f"Constructing {len(regions)} region analyzers for {sample.sample_id} "
            )
            yield (
                sample.sample_id,
                [
                    ra.RegionAnalyzer.fromRegion(
                        description.getRegion(region_name),
                        sample,
                        MODULE_REPO,
                        era_repo,
                    )
                    for region_name in regions
                ],
            )
