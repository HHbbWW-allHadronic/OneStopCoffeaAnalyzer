from __future__ import annotations
import json
import functools as ft
import time
from typing import Any
import operator as op
from attrs import define, field
import base64
from analyzer.core.results import ResultBase
import os
from distributed import Client, LocalCluster, as_completed
from .executor import Executor, CompletedTask
from analyzer.core.event_collection import FileInfo
from rich import print

# import analyzer.core.dask_sizes  # noqa
import math
import logging
from dask.sizeof import sizeof


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


logger = logging.getLogger(__name__)


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
                reduced_futures = reduceResults(
                    client,
                    iaddMany,
                    task_futures,
                    reduction_factor,
                    key_suffix=f"{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
                )
                final = client.map(
                    ft.partial(dumpAndComplete, task.metadata, task.output_name),
                    reduced_futures,
                    key=f"complete--{task.metadata['dataset_name']}-{task.metadata['sample_name']}",
                )
                as_comp.update(final)

        elif isinstance(result, DaskRunResult):
            ret = result.maybe_result
            exceptions = result.maybe_exceptions
            if ret is not None:
                yield ret
            print(exceptions)
            future.cancel()


@define
class DaskExecutor(Executor):
    def setup(self, needed_resources):
        if self.adapt:
            self._cluster.adapt(
                minimum_jobs=self.min_workers, maximum_jobs=self.max_workers
            )

    def teardown(self):
        self._client.close()
        self._cluster.close()

    def run(self, *args, **kwargs):
        self.runFutured(*args, **kwargs)
        return


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
        # with open(CONFIG.DASK_CONFIG_PATH) as f:
        #     defaults = yaml.safe_load(f)
        #     dask.config.update(dask.config.config, defaults, priority="new")
        #
        # log_config_path = Path(CONFIG.CONFIG_PATH) / "worker_logging_config.yaml"
        # with open(log_config_path, "rt") as f:
        #     config = yaml.safe_load(f.read())
        #     c = base64.b64encode(json.dumps(config).encode("utf-8")).decode("utf-8")
        # os.environ["ANALYZER_LOG_CONFIG"] = c
        # self.cluster = LocalCluster(
        #     dashboard_address=self.dashboard_address,
        #     memory_limit=self.worker_memory,
        #     n_workers=self.max_workers,
        #     scheduler_kwargs={"host": self.schedd_address},
        #     processes=self.processes,
        # )

        # self.client = Client(self.cluster)
        self.client = Client("tcp://192.168.11.48:8786")

    def run(self, analyzer, tasks):
        yield from run(
            self.client, self.chunk_size, self.reduction_factor, analyzer, tasks
        )


@define
class LPCCondorDask(Executor):
    apptainer_working_dir: str | None = None
    base_working_dir: str | None = None
    venv_path: str | None = None
    x509_path: str | None = None
    base_condor_log_location: str | None = "/uscmst1b_scratch/lpc1/3DayLifetime/"
    temporary_path: str | None = ".application_data/temporary"
    worker_timeout: int | None = 7200
    analysis_root_dir: str | None = "/srv"

    def setup(self):
        from lpcjobqueue import LPCCondorCluster
        import shutil

        self.venv_path = self.venv_path or os.environ.get("VIRTUAL_ENV")
        self.x509_path = self.x509_path or os.environ.get("X509_USER_PROXY")
        self.apptainer_working_dir = self.apptainer_working_dir or os.environ.get(
            "APPTAINER_WORKING_DIR", "."
        )
        self._working_dir = str(Path(".").absolute())

        with open(CONFIG.DASK_CONFIG_PATH) as f:
            defaults = yaml.safe_load(f)
            dask.config.update(dask.config.config, defaults, priority="new")
        apptainer_container = "/".join(
            Path(os.environ["APPTAINER_CONTAINER"]).parts[-2:]
        )
        Path(os.environ.get("APPTAINER_WORKING_DIR", ".")).resolve()

        logpath = Path(self.base_log_path) / os.getlogin() / "dask_logs"
        if logpath.exists():
            logger.info("Deleting old dask logs")
            shutil.rmtree(logpath, ignore_errors=True)
        logpath.mkdir(exist_ok=True, parents=True)

        transfer_input_files = setupForCondor(
            analysis_root_dir=self._working_dir,
            apptainer_dir=self.apptainer_working_dir,
            venv_path=self.venv_path,
            x509_path=self.x509_path,
            temporary_path=self.temporary_path,
            extra_files=self.extra_files,
        )
        kwargs = {}
        kwargs["worker_extra_args"] = [
            *dask.config.get("jobqueue.lpccondor.worker_extra_args"),
            "--preload",
            "lpcjobqueue.patch",
            "--preload",
            "analyzer.core.dask_sizes",
        ]
        kwargs["job_extra_directives"] = {"+MaxRuntime": self.worker_timeout}
        kwargs["python"] = f"{str(self.venv_path)}/bin/python"
        prologue = dask.config.get("jobqueue.lpccondor.job-script-prologue")
        prologue.append(
            "export DASK_INTERNAL_INHERIT_CONFIG="
            + dask.config.serialize(dask.config.global_config)
        )
        log_config_path = Path(CONFIG.CONFIG_PATH) / "worker_logging_config.yaml"
        with open(log_config_path, "r") as f:
            config = yaml.safe_load(f.read())
            c = base64.b64encode(json.dumps(config).encode("utf-8")).decode("utf-8")
        prologue.append("export ANALYZER_LOG_CONFIG=" + c)

        logger.info(f"Transfering input files: \n{transfer_input_files}")
        self._cluster = LPCCondorCluster(
            ship_env=False,
            image=apptainer_container,
            memory=self.worker_memory,
            transfer_input_files=transfer_input_files,
            log_directory=logpath,
            scheduler_options=dict(dashboard_address=self.dashboard_address),
            job_script_prologue=prologue,
            **kwargs,
        )
        self._client = Client(self._cluster)
        try:
            os.system("ulimit -n 4096")
        except Exception:
            pass
        super().setup()
