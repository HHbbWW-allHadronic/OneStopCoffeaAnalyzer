from __future__ import annotations

import abc
from attrs import define
from analyzer.core.event_collection import FileSet



@define
class ExecutionTask:
    event_source_desc: FileSet
    metadata: dict

@define
class Executor(abc.ABC):

    @abc.abstractmethod
    def run(self, tasks: ExecutionTasks):
        pass

    def setup(self, needed_resources):
        pass

    def teardown(self):
        pass

    def __exit__(self, type, value, traceback):
        self.teardown()

    def __enter__(self):
        self.setup()
