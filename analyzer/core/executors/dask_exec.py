from __future__ import annotations
from analyzer.configuration import CONFIG
import time
from pathlib import Path
import json
from distributed.diagnostics.plugin import UploadDirectory
import functools as ft
import dask
import time
from typing import Any
import operator as op
from attrs import define, field
import base64
from analyzer.core.results import ResultBase
import os
from distributed import Client, LocalCluster, as_completed
from .executor import Executor, CompletedTask
from .condor_tools import createCondorPackage
from analyzer.core.event_collection import FileInfo
from rich import print
import tempfile
import uuid
import zipfile

# import analyzer.core.dask_sizes  # noqa
import math
import logging
from dask.sizeof import sizeof
import os


class AnalyzerRuntimeError(ExceptionGroup):
    def derive(self, excs):
        return AnalyzerRuntimeError(self.message, excs)


@define
class DaskRunException:
    chunk: FileChunk
    exception: Exception


@define
class DaskRunResult:
    maybe_result: ResultBase | None
    maybe_exceptions: list[DaskRunException] = field(factory=list)

    def __iadd__(self, other):
        if self.maybe_result is None and other.maybe_result is not None:
            self.maybe_result = other.maybe_result
        elif self.maybe_result is not None and other.maybe_result is not None:
            self.maybe_result += other.maybe_result
        self.maybe_exceptions += other.maybe_exceptions
        return self


@sizeof.register(DaskRunResult)
def _(obj):
    ret = sizeof(obj.maybe_result)
    return ret


@sizeof.register(CompletedTask)
def _(obj):
    ret = sizeof(obj.result)
    return ret


@sizeof.register(ResultBase)
def _(obj):
    ret = obj.approxSize()
    return ret


def iaddMany(to_add):
    ret = to_add[0]
    for x in to_add[1:]:
        ret += x
    return ret


def reduceResults(
    client,
    reduction_function,
    futures,
    reduction_factor=5,
    target_final_count=1,
    close_to_target_frac=0.8,
    key_suffix="",
):
    layer = 0
    while len(futures) > target_final_count:
        if (len(futures) / reduction_factor) < (
            target_final_count * close_to_target_frac
        ):
            reduction_factor = math.ceil(len(futures) / target_final_count)
        futures = [
            client.submit(
                reduction_function,
                futures[i : i + reduction_factor],
                key=f"merge-{layer}-{i}" + str(key_suffix),
            )
            for i in range(0, len(futures), reduction_factor)
        ]

        layer += 1
    return futures


logger = logging.getLogger("analyzer")


def getAnalyzerRunFunc(analyzer, task):
    def inner(chunk):
        try:
            return DaskRunResult(analyzer.run(chunk, task.metadata, task.pipelines), [])
        except Exception as e:
            return DaskRunResult(None, [DaskRunException(chunk, e)])

    return inner


def dumpAndComplete(metadata, output_name, dask_result):
    result, exceptions = dask_result.maybe_result, dask_result.maybe_exceptions
    if result is not None:
        result = result.toBytes()
    return DaskRunResult(CompletedTask(result, metadata, output_name), exceptions)


def run(client, chunk_size, reduction_factor, analyzer, tasks):
    tasks = {i: x for i, x in enumerate(tasks)}

    file_prep_tasks = {}
    file_prep_task_mapping = {}

    for i, task in tasks.items():
        file_set = task.file_set
        file_set.updateFromCache()
        needed_updates = file_set.getNeededUpdatesFuncs()
        futures = client.map(
            lambda x: x(),
            needed_updates,
            key=f"prep--{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
        )

        file_prep_tasks[i] = set(futures)
        for f in futures:
            file_prep_task_mapping[f] = i

    as_comp = as_completed((y for x in file_prep_tasks.values() for y in x))

    for future in as_comp:
        try:
            result = future.result()
        except Exception as e:
            result = None
        if future in file_prep_task_mapping:
            index = file_prep_task_mapping.pop(future)
            if result is not None:
                tasks[index].file_set.updateFileInfo(result)
            future.cancel()
            file_prep_tasks[index].remove(future)
            if not file_prep_tasks[index]:
                task = tasks[index]
                task_futures = client.map(
                    getAnalyzerRunFunc(analyzer, task),
                    list(task.file_set.toChunked(chunk_size).iterChunks()),
                    key=f"process--{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
                )
                with dask.annotate(priority=10):
                    reduced_futures = reduceResults(
                        client,
                        iaddMany,
                        task_futures,
                        reduction_factor,
                        key_suffix=f"{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
                    )
                with dask.annotate(priority=20):
                    final = client.map(
                        ft.partial(dumpAndComplete, task.metadata, task.output_name),
                        reduced_futures,
                        key=f"complete--{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
                    )
                as_comp.update(final)

        elif isinstance(result, DaskRunResult):
            if result is None:
                continue
            ret = result.maybe_result
            exceptions = result.maybe_exceptions
            if ret is not None:
                yield ret
            print(exceptions)
            future.cancel()


@define
class LocalDaskExecutor(Executor):
    max_workers: int
    min_workers: int

    worker_memory: str = "4GB"
    dashboard_address: str | None = "localhost:8789"
    schedd_address: str | None = "localhost:12358"
    adapt: bool = True
    chunk_size: int | None = 100000
    processes: bool = True
    cluster: Any = None
    client: Any = None
    reduction_factor: int = 10

    def setup(self, needed_resources):
        self.cluster = LocalCluster(
            dashboard_address=self.dashboard_address,
            memory_limit=self.worker_memory,
            n_workers=self.max_workers,
            scheduler_kwargs={"host": self.schedd_address},
            processes=self.processes,
        )

        self.client = Client(self.cluster)

    def run(self, analyzer, tasks):
        yield from run(
            self.client, self.chunk_size, self.reduction_factor, analyzer, tasks
        )


@define
class LPCCondorDask(Executor):
    container: str
    venv_path: str | None = None
    x509_path: str | None = None
    log_path: str = "logs/condor"
    worker_timeout: int | None = 7200
    min_workers: int = 1
    max_workers: int = 10

    worker_memory: str = "2GB"
    dashboard_address: str | None = "localhost:8789"
    schedd_address: str | None = "localhost:12358"
    adapt: bool = True
    chunk_size: int | None = 100000
    reduction_factor: int = 10
    cluster: Any = None
    client: Any = None

    def run(self, analyzer, tasks):
        yield from run(
            self.client, self.chunk_size, self.reduction_factor, analyzer, tasks
        )

    def setup(self, needed_resources):
        import shutil

        condor_temp_loc = (
            Path(CONFIG.general.base_data_path) / CONFIG.condor.temp_location
        )
        condor_config = condor_temp_loc / ".cmslpc-local-conf"
        # os.environ["LPC_CONDOR_CONFIG"] = "/etc/condor/config.d/01_cmslpc_interactive"
        #         os.environ["LPC_CONDOR_LOCAL"] = str(condor_config)
        # os.environ["CONDOR_CONFIG"] = os.environ["LPC_CONDOR_CONFIG"]
        #
        #
        #         if not condor_config.exists():
        #             with open(condor_config, "w") as f:
        #                 f.write(
        #                     """#!/bin/bash
        # python3 /usr/local/bin/cmslpc-local-conf.py | grep -v "LOCAL_CONFIG_FILE"""
        #                 )
        #         breakpoint()
        from lpcjobqueue import LPCCondorCluster

        package = createCondorPackage(self.container, self.venv_path)
        logpath = Path(self.log_path).resolve()
        logpath.mkdir(exist_ok=True, parents=True)
        kwargs = {}
        kwargs["worker_extra_args"] = [
            *dask.config.get("jobqueue.lpccondor.worker_extra_args")
        ]
        kwargs["job_extra_directives"] = {
            "+MaxRuntime": self.worker_timeout,
        }
        kwargs["python"] = f"{str(self.venv_path)}/bin/python"
        self.cluster = LPCCondorCluster(
            ship_env=False,
            image=package.container,
            memory=self.worker_memory,
            transfer_input_files=package.transfer_file_list,
            log_directory=logpath,
            scheduler_options=dict(dashboard_address=self.dashboard_address),
            job_script_prologue=["source setup.sh"],
            **kwargs,
        )
        self.cluster.adapt(minimum_jobs=self.min_workers, maximum_jobs=self.max_workers)
        self.client = Client(self.cluster)


if __name__ == "__main__":
    main()
