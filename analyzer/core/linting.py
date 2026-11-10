from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING
import abc
from attrs import define, field
from collections import defaultdict

if TYPE_CHECKING:
    from analyzer.core.analysis import Analysis
    from analyzer.core.analysis_modules import AnalyzerModule

logger = logging.getLogger("analyzer.core.linting")


class LintLevel(Enum):
    WARNING = 1
    ERROR = 2


@define
class LintMessage:
    level: LintLevel
    category: str
    message: str
    pipeline_name: str | None = None
    module_name: str | None = None

    def __str__(self):
        ctx = []
        if self.pipeline_name:
            ctx.append(f"Pipeline: {self.pipeline_name}")
        if self.module_name:
            ctx.append(f"Module: {self.module_name}")
        ctx_str = f" [{', '.join(ctx)}]" if ctx else ""
        return f"[{self.level.name}] {self.category}{ctx_str}: {self.message}"


@define
class LintConfig:
    ignore: list[str] = field(factory=list)


class PipelineLinter(abc.ABC):
    name: str = "BasePipelineLinter"

    @abc.abstractmethod
    def lint(
        self, pipeline_name: str, pipeline: list[AnalyzerModule], metadatas: list[dict]
    ) -> list[LintMessage]:
        return []


class AnalysisLinter(abc.ABC):
    name: str = "BaseAnalysisLinter"

    @abc.abstractmethod
    def lint(self, analysis: Analysis) -> list[LintMessage]:
        return []


class DanglingSelectionLinter(PipelineLinter):
    name = "DanglingSelection"

    def lint(
        self, pipeline_name: str, pipeline: list[AnalyzerModule], metadatas: list[dict]
    ) -> list[LintMessage]:
        from analyzer.modules.common.selection import SelectOnColumns

        messages = []
        pending_selections = set()
        meta = metadatas[0]

        for module in pipeline:
            outputs = module.outputs(meta)
            if outputs != "EVENTS":
                for col in outputs:
                    if len(col.fields) >= 2 and col.fields[0] == "Selection":
                        pending_selections.add(col.fields[1])

            if isinstance(module, SelectOnColumns):
                if module.selection_names is not None:
                    for s in module.selection_names:
                        pending_selections.discard(s)
                else:
                    pending_selections.clear()

        for sel in pending_selections:
            messages.append(
                LintMessage(
                    level=LintLevel.WARNING,
                    category=self.name,
                    message=f"Selection '{sel}' was added but never consumed by a SelectOnColumns module.",
                    pipeline_name=pipeline_name,
                )
            )

        return messages


class DataQualityLinter(PipelineLinter):
    name = "DataQuality"

    def lint(
        self, pipeline_name: str, pipeline: list[AnalyzerModule], metadatas: list[dict]
    ) -> list[LintMessage]:
        from analyzer.core.analysis_modules import IsSampleType

        messages = []

        runs_on_data = False
        expr = IsSampleType("Data")
        for meta in metadatas:
            if expr.evaluate(meta):
                runs_on_data = True
                break

        if not runs_on_data:
            return messages

        has_golden = any(type(m).__name__ == "GoldenLumi" for m in pipeline)
        has_noise = any(type(m).__name__ == "NoiseFilter" for m in pipeline)

        if not has_golden:
            messages.append(
                LintMessage(
                    level=LintLevel.WARNING,
                    category=self.name,
                    message="Pipeline is missing a 'GoldenLumi' module. If this pipeline runs on Data, it will not apply golden JSON certification.",
                    pipeline_name=pipeline_name,
                )
            )
        if not has_noise:
            messages.append(
                LintMessage(
                    level=LintLevel.WARNING,
                    category=self.name,
                    message="Pipeline is missing a 'NoiseFilter' module. If this pipeline runs on Data, it will not apply standard noise filters.",
                    pipeline_name=pipeline_name,
                )
            )

        return messages


class PipelineReferenceLinter(AnalysisLinter):
    name = "PipelineReference"

    def lint(self, analysis: Analysis) -> list[LintMessage]:
        messages = []
        defined_pipelines = set(analysis.analyzer.base_pipelines.keys())
        for desc in analysis.event_collections:
            for p in desc.pipelines:
                if p not in defined_pipelines:
                    messages.append(
                        LintMessage(
                            level=LintLevel.ERROR,
                            category=self.name,
                            message=f"Event collection references undefined pipeline '{p}'.",
                        )
                    )
        return messages


class MissingCutflowLinter(PipelineLinter):
    name = "MissingCutflow"

    def lint(
        self, pipeline_name: str, pipeline: list[AnalyzerModule], metadatas: list[dict]
    ) -> list[LintMessage]:
        from analyzer.modules.common.selection import SelectOnColumns

        messages = []
        has_cutflow = False
        has_selection = False

        for module in pipeline:
            if isinstance(module, SelectOnColumns):
                has_selection = True
                if module.save_cutflow:
                    has_cutflow = True

        if has_selection and not has_cutflow:
            messages.append(
                LintMessage(
                    level=LintLevel.WARNING,
                    category=self.name,
                    message="Pipeline applies selections but no 'SelectOnColumns' module has 'save_cutflow=True'. Cutflow yields will not be saved.",
                    pipeline_name=pipeline_name,
                )
            )
        return messages


PIPELINE_LINTERS: list[PipelineLinter] = [
    DanglingSelectionLinter(),
    DataQualityLinter(),
    MissingCutflowLinter(),
]

ANALYSIS_LINTERS: list[AnalysisLinter] = [
    PipelineReferenceLinter(),
]


def runLint(analysis: Analysis) -> list[LintMessage]:
    from analyzer.core.running import getRepos, getTasks

    all_messages = []
    dataset_repo, era_repo = getRepos(
        analysis.extra_dataset_paths, analysis.extra_era_paths
    )
    tasks = getTasks(dataset_repo, era_repo, analysis.event_collections)
    pipeline_metadatas = defaultdict(list)
    for task in tasks:
        for pipeline in task.pipelines:
            pipeline_metadatas[pipeline].append(task.metadata)

    for linter in ANALYSIS_LINTERS:
        msgs = linter.lint(analysis)
        all_messages.extend(msgs)

    for pipeline_name, pipeline in analysis.analyzer.base_pipelines.items():
        metadatas = pipeline_metadatas.get(pipeline_name, [])
        for module in pipeline:
            msgs = module.lint()
            for msg in msgs:
                msg.pipeline_name = pipeline_name
            all_messages.extend(msgs)

        for linter in PIPELINE_LINTERS:
            msgs = linter.lint(pipeline_name, pipeline, metadatas)
            all_messages.extend(msgs)

    filtered_messages = []
    for msg in all_messages:
        if msg.category not in analysis.lint_config.ignore:
            filtered_messages.append(msg)

    return filtered_messages
