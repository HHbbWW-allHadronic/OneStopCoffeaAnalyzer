from __future__ import annotations

import abc
from attrs import define

@define
class Executor(abc.ABC):

    @abc.abstractmethod
    def run(self, tasks, result_complete_callback=None):
        pass

    def setup(self, needed_resources):
        pass

    def teardown(self):
        pass

    def __exit__(self, type, value, traceback):
        self.teardown()

    def __enter__(self):
        self.setup()
