from __future__ import annotations

import abc
from typing import Any


from pydantic import BaseModel


class Executor(abc.ABC, BaseModel):
    test_mode: bool = False


    @abc.abstractmethod
    def run(self, tasks: list[AnalysisTask], result_complete_callback=None):
        pass

    def setup(self, needed_resources):
        pass

    def teardown(self):
        pass

    def __exit__(self, type, value, traceback):
        self.teardown()

    def __enter__(self):
        self.setup()
