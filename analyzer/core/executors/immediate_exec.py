from __future__ import annotations

import copy
from typing import Literal


import analyzer.core.results as core_results
from analyzer.utils.file_tools import extractCmsLocation
from analyzer.utils.structure_tools import accumulate
from analyzer.core.event_collection import getFileEvents
from .executor import Executor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from .preprocessing_tools import preprocess
import logging

logger = logging.getLogger(__name__)


class ImmediateExecutor(Executor):
    chunk_size: int = 100000

    def run(self, analyzer, tasks):
        for t in tasks:
            file_set = task.file_set
            file_set.chunk_size = file_set.chunk_size or self.chunk_size
            for func in file_set.getNeededUpdatesFuncs():
                file_set.updateEvents(*func())

            chunks = list(file_set.iterChunks())
            for chunk in chunks:
                yield analyzer.run(chunk, task.pipelines)

    

