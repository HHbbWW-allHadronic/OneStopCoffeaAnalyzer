from __future__ import annotations

import copy
from .finalizers import basicFinalizer
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

    def run(self, analyzer, tasks, max_sample_events=None):
        orig_analyzer =  copy.deepcopy(analyzer)
        for task in tasks:
            logger.info(f"Running task with output {task.output_name}")
            file_set = task.file_set
            file_set.updateFromCache()
            needed_updates = file_set.getNeededUpdatesFuncs()
            for update in needed_updates:
                u = update()
                file_set.updateFileInfo(update())
                if max_sample_events and file_set.total_file_events >= max_sample_events:
                    break
            chunked = file_set.toChunked(max_sample_events or self.chunk_size)
            for chunk in chunked.iterChunks():
                analyzer = copy.deepcopy(orig_analyzer)
                result = analyzer.run(chunk, task.metadata, task.pipelines)
                result.finalize(basicFinalizer)
                yield CompletedTask(result.toBytes(), task.metadata, task.output_name)
                if max_sample_events:
                     break
