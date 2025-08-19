from __future__ import annotations

import logging


import analyzer.core.analyzer as core_analyzer
from analyzer.datasets import FileSet, SampleId, SampleParams
from pydantic import BaseModel
from analyzer.core.executors.types import AnyExecutor

logger = logging.getLogger(__name__)


class AnalysisTask(BaseModel):
    sample_id: SampleId
    sample_params: SampleParams
    file_set: FileSet
    analyzer: core_analyzer.Analyzer
    

class PackagedTask(BaseModel):
    identifier: str
    executor: AnyExecutor
    task: AnalysisTask
