from __future__ import annotations

import copy
from typing import Literal
from analyzer.core.event_collection import getFileEvents
from .executor import Executor, CompletedTask
import logging
from attrs import define

logger = logging.getLogger(__name__)


@define
class ImmediateExecutor(Executor):
    chunk_size: int = 100000

    def run(self, analyzer, tasks):
        for task in tasks:
            file_set = task.file_set
            file_set.chunk_size = file_set.chunk_size or self.chunk_size
            for func in file_set.getNeededUpdatesFuncs():
                file_set.updateEvents(*func())

            chunked = file_set.toChunked(self.chunk_size)
            for chunk in chunked.iterChunks():
                result = analyzer.run(chunk, task.metadata, task.pipelines)
                yield CompletedTask(result.toBytes(), task.metadata, task.output_name)
