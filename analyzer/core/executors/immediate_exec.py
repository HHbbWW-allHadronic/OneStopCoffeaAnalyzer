from __future__ import annotations

import copy
from typing import Literal
from analyzer.core.event_collection import FileInfo
from .executor import Executor, CompletedTask
import logging
from attrs import define
from analyzer.core.event_collection import getFileInfo

logger = logging.getLogger("analyzer")


@define
class ImmediateExecutor(Executor):
    chunk_size: int = 100000

    def run(self, analyzer, tasks):
        for task in tasks:
            file_set = task.file_set
            file_set.updateFromCache()
            needed_updates = file_set.getNeededUpdatesFuncs()
            for update in needed_updates:
                file_set.updateFileInfo(update())
            chunked = file_set.toChunked(self.chunk_size)
            for chunk in chunked.iterChunks():
                result = analyzer.run(chunk, task.metadata, task.pipelines)
                yield CompletedTask(result.toBytes(), task.metadata, task.output_name)
