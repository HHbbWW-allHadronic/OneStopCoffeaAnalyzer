from __future__ import annotations

from typing import Annotated


from pydantic import Field

from analyzer.core.executors.dask_exec import LocalDaskExecutor, LPCCondorDask
from analyzer.core.executors.condor_exec import CondorExecutor
from analyzer.core.executors.immediate_exec import ImmediateExecutor


AnyExecutor = Annotated[
    LocalDaskExecutor | CondorExecutor | ImmediateExecutor | LPCCondorDask,
    Field(discriminator="executor_type"),
]
