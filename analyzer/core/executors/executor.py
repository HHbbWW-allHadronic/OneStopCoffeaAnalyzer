from __future__ import annotations

import abc
from typing import Any


from pydantic import BaseModel



class Executor(abc.ABC, BaseModel):
    test_mode: bool = False

    def setup(self):
        pass

    @abc.abstractmethod
    def run(self, tasks: dict[Any, AnalysisTask], result_complete_callback=None):
        pass
