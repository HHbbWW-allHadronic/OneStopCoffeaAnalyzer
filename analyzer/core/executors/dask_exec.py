from __future__ import annotations
import copy

import functools as ft
import logging

# import analyzer.core.dask_sizes  # noqa
import math
from pathlib import Path
from typing import Any

import dask
from analyzer.configuration import CONFIG
from analyzer.core.event_collection import FileChunk
from analyzer.core.results import ResultBase
from attrs import define, field
from dask.sizeof import sizeof
from distributed import Client, LocalCluster, as_completed
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .condor_tools import createCondorPackage
from .executor import CompletedTask, Executor
from .finalizers import basicFinalizer
from concurrent.futures import ProcessPoolExecutor, TimeoutError


def configureDask():
    from analyzer import static
    import importlib.resources
    import yaml

    default_dask_config_path = importlib.resources.files(static) / "dask_config.yaml"

    with open(default_dask_config_path) as f:
        defaults = yaml.safe_load(f)
        dask.config.update(dask.config.config, defaults, priority="new")


def callTimeout(process_timeout, function, *args, **kwargs):
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(function, *args, **kwargs)
            return future.result(timeout=process_timeout)
        except TimeoutError:
            for _, process in executor._processes.items():
                process.terminate()
            raise


class AnalyzerRuntimeError(ExceptionGroup):
    def derive(self, excs):
        return AnalyzerRuntimeError(self.message, excs)


@define
class DaskRunException:
    chunk: FileChunk
    exception: Exception


@define
class DaskRunResult:
    maybe_result: CompletedTask | ResultBase | None
    maybe_exceptions: list[DaskRunException] = field(factory=list)
    events_processed: int = 0

    def __iadd__(self, other: DaskRunResult):
        if self.maybe_result is None and other.maybe_result is not None:
            self.maybe_result = other.maybe_result
        elif self.maybe_result is not None and other.maybe_result is not None:
            self.maybe_result += other.maybe_result
        self.maybe_exceptions += other.maybe_exceptions
        self.events_processed += other.events_processed
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


def runWithFinalize(analyzer, *args, **kwargs):
    analyzer = copy.deepcopy(analyzer)
    ret = analyzer.run(*args, **kwargs)
    ret.finalize(basicFinalizer)
    return ret


def getAnalyzerRunFunc(analyzer, task, timeout=120):
    def inner(chunk):
        try:
            ret = callTimeout(
                timeout, runWithFinalize, analyzer, chunk, task.metadata, task.pipelines
            )
            # ret = runWithFinalize(analyzer, chunk, task.metadata, task.pipelines)
            return DaskRunResult(ret, [], chunk.nevents)
        except Exception as e:
            return DaskRunResult(None, [DaskRunException(chunk, e)], chunk.nevents)

    return inner


def dumpAndComplete(metadata, output_name, dask_result):
    result, exceptions = dask_result.maybe_result, dask_result.maybe_exceptions
    if result is not None:
        result = result.toBytes()

    return DaskRunResult(
        CompletedTask(result, metadata, output_name),
        exceptions,
        dask_result.events_processed,
    )


def processTask(
    client,
    analyzer,
    task,
    chunk_size,
    reduction_factor,
    max_sample_events=None,
    timeout=120,
):
    chunked = task.file_set.toChunked(chunk_size)
    n_events = chunked.chunked_events
    chunks = list(chunked.iterChunks())
    if max_sample_events:
        new_chunks = []
        total = 0
        for c in chunks:
            total += c.nevents or 0
            new_chunks.append(c)
            if total > max_sample_events:
                break

        chunks = new_chunks

    with dask.annotate(priority=0):
        task_futures = client.map(
            getAnalyzerRunFunc(analyzer, task, timeout=timeout),
            chunks,
            key=f"analyze--{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
        )
    with dask.annotate(priority=0):
        reduced_futures = reduceResults(
            client,
            iaddMany,
            task_futures,
            reduction_factor,
            key_suffix=f"{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
        )
    with dask.annotate(priority=3):
        final = client.map(
            ft.partial(dumpAndComplete, task.metadata, task.output_name),
            reduced_futures,
            key=f"complete--{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
        )
    return n_events, final


def run(
    client,
    chunk_size,
    reduction_factor,
    analyzer,
    tasks,
    max_sample_events=None,
    timeout=120,
):
    tasks = {i: x for i, x in enumerate(tasks)}

    file_prep_tasks = {}
    file_prep_task_mapping = {}

    progress_bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    bar_prep = progress_bar.add_task("Prep Tasks")
    bar_events = progress_bar.add_task("Events")
    total_prep = 0
    total_events = 0

    for i, task in tasks.items():
        file_set = task.file_set
        file_set.updateFromCache()
        needed_updates = file_set.getNeededUpdatesFuncs()
        futures = client.map(
            lambda x: x(),
            needed_updates,
            key=f"prep--{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
        )
        total_prep += len(needed_updates)
        file_prep_tasks[i] = set(futures)
        for f in futures:
            file_prep_task_mapping[f] = i

    progress_bar.update(bar_prep, total=total_prep)
    progress_bar.update(bar_events, total=0)

    as_comp = as_completed((y for x in file_prep_tasks.values() for y in x))

    for i, task in tasks.items():
        if not file_prep_tasks[i]:
            n_events, new_tasks = processTask(
                client,
                analyzer,
                task,
                chunk_size,
                reduction_factor,
                max_sample_events=max_sample_events,
                timeout=timeout,
            )
            as_comp.update(new_tasks)
            total_events += n_events

    progress_bar.update(bar_events, total=total_events)

    with progress_bar:
        for future in as_comp:
            try:
                result = future.result()
            except Exception as e:
                logger.warning(e)
                result = None
            if future in file_prep_task_mapping:
                progress_bar.update(bar_prep, advance=1)
                index = file_prep_task_mapping.pop(future)
                if result is not None:
                    tasks[index].file_set.updateFileInfo(result)
                future.cancel()
                file_prep_tasks[index].remove(future)
                if not file_prep_tasks[index]:
                    task = tasks[index]
                    n_events, new_tasks = processTask(
                        client,
                        analyzer,
                        task,
                        chunk_size,
                        reduction_factor,
                        max_sample_events=max_sample_events,
                    )
                    as_comp.update(new_tasks)
                    total_events += n_events
                    progress_bar.update(bar_events, total=total_events)

            elif isinstance(result, DaskRunResult):
                ret = result.maybe_result
                progress_bar.update(bar_events, advance=result.events_processed)
                if ret.result is not None:
                    yield ret
                else:
                    logger.warning(
                        f"Result was None. Encountered exceptions during execution:\n{result.maybe_exceptions}"
                    )

                future.cancel()


@define
class LocalDaskExecutor(Executor):
    max_workers: int
    min_workers: int

    worker_memory: str = "4GB"
    dashboard_address: str = "localhost:8789"
    schedd_address: str | None = "localhost:12358"
    adapt: bool = True
    chunk_size: int | None = 100000
    processes: bool = True
    cluster: Any = None
    client: Any = None
    reduction_factor: int = 10
    timeout: int = 120

    def setup(self, needed_resources):
        configureDask()
        self.cluster = LocalCluster(
            dashboard_address=self.dashboard_address,
            memory_limit=self.worker_memory,
            n_workers=self.max_workers,
            scheduler_kwargs={"host": self.schedd_address},
            processes=self.processes,
        )

        self.client = Client(self.cluster)

    def run(self, analyzer, tasks, max_sample_events=None):
        with self.cluster:
            yield from run(
                self.client,
                self.chunk_size,
                self.reduction_factor,
                analyzer,
                tasks,
                max_sample_events=max_sample_events,
                timeout=self.timeout,
            )


@define
class LPCCondorDask(Executor):
    container: str
    venv_path: str = ".venv"
    x509_path: str | None = None
    log_path: str = "logs/condor"
    worker_timeout: int | None = 7200
    min_workers: int = 1
    max_workers: int = 10

    worker_memory: str = "4GB"
    dashboard_address: str | None = "localhost:8789"
    schedd_address: str | None = "localhost:12358"
    adapt: bool = True
    chunk_size: int | None = 100000
    reduction_factor: int = 3
    timeout: int = 120
    cluster: Any = None
    client: Any = None

    def run(self, analyzer, tasks, max_sample_events=None):
        with self.cluster:
            yield from run(
                self.client,
                self.chunk_size,
                self.reduction_factor,
                analyzer,
                tasks,
                max_sample_events=max_sample_events,
                timeout=self.timeout,
            )

    def setup(self, needed_resources):
        configureDask()

        condor_temp_loc = (
            Path(CONFIG.general.base_data_path) / CONFIG.condor.temp_location
        )
        condor_temp_loc / ".cmslpc-local-conf"
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

        package = createCondorPackage(self.container, self.venv_path, needed_resources)
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
        # prologue = dask.config.get("jobqueue.lpccondor.job-script-prologue")
        # prologue.append(
        #     "export DASK_INTERNAL_INHERIT_CONFIG="
        #     + dask.config.serialize(dask.config.global_config)
        # )

        prologue = ["export DASK_DISTRIBUTED__WORKER__DAEMON=0", "source setup.sh"]

        self.cluster = LPCCondorCluster(
            ship_env=False,
            image=package.container,
            memory=self.worker_memory,
            transfer_input_files=package.transfer_file_list,
            log_directory=logpath,
            scheduler_options=dict(dashboard_address=self.dashboard_address),
            job_script_prologue=prologue,
            **kwargs,
        )
        logger.info(f"Started cluster {self.cluster}")
        self.cluster.adapt(minimum_jobs=self.min_workers, maximum_jobs=self.max_workers)
        self.client = Client(self.cluster)
