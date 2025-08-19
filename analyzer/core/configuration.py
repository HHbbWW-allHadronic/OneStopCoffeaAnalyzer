import enum
from collections import defaultdict
import re
from typing import Any, ClassVar

import yaml

from pydantic import BaseModel, Field

from .analysis_modules import MODULE_REPO

import logging

import analyzer.core.specifiers as specs
import analyzer.core.region_analyzer as ra
import analyzer.core.executors as executors

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
    sample_spec: specs.SampleSpec | None = None
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


class AnalysisDescription(BaseModel):
    name: str
    executors: dict[str, executors.AnyExecutor]
    # execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    file_config: FileConfig = Field(default_factory=FileConfig)
    samples: dict[str, list[str] | str]
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


def getSubSectors(description, dataset_repo, era_repo):
    s_pairs = []
    ret = defaultdict(list)
    for dataset_name, regions in description.samples.items():
        if any(x in dataset_name for x in [".", "*"]):
            todo = [(x, regions) for x in dataset_repo if re.match(dataset_name, x)]
        else:
            todo = [(dataset_name, regions)]
        for dataset_name, regions in todo:
            if isinstance(regions, str) and regions == "All":
                regions = [r.name for r in description.regions]
            for r in regions:
                s_pairs.append((dataset_name, r))
    for dataset_name, region_name in s_pairs:
        logger.debug(
            f'Getting analyzer for dataset "{dataset_name}" and region "{region_name}"'
        )
        dataset = dataset_repo[dataset_name]
        region = description.getRegion(region_name)
        for sample in dataset.samples:
            subsector = ra.RegionAnalyzer.fromRegion(
                region,
                sample,
                MODULE_REPO,
                era_repo,
            )
            ret[sample.sample_id].append(subsector)

    return ret
