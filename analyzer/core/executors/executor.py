from __future__ import annotations

import abc
import functools as ft 
from attrs import define
from typing import Any
from analyzer.core.event_collection import FileSet
from cattrs.strategies import include_subclasses, configure_tagged_union



@define
class ExecutionTask:
    file_set: FileSet
    metadata: dict
    pipelines: list[str]
    output_name: str

@define
class CompletedTask:
    result: Any
    metadata: dict
    output_name: str

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

def configureConverter(conv):
    union_strategy = ft.partial(configure_tagged_union, tag_name="executor_name")
    include_subclasses(Executor, conv, union_strategy=union_strategy)
