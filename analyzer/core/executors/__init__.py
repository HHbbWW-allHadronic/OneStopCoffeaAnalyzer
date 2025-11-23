from __future__ import annotations

from .executor import Executor, ExecutionTask
import  analyzer.core.executors.immediate_exec
import  analyzer.core.executors.dask_exec
from  analyzer.core.executors.premade_executors import getPremadeExcutors


