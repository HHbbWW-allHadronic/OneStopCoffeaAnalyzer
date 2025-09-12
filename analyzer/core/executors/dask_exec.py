from __future__ import annotations
import json
import lz4.frame
import pickle as pkl
import cProfile
import time
import base64
import os

import concurrent.futures
import gc
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import yaml
import uproot

import analyzer.core.results as core_results
import awkward as ak
import dask
from analyzer.configuration import CONFIG
import math
from analyzer.utils.structure_tools import iadd
from analyzer.datasets import FileSet
from analyzer.core.exceptions import AnalysisRuntimeError
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import compress_form, decompress_form
from distributed import (
    Client,
    LocalCluster,
    as_completed,
    get_client,
    secede,
    rejoin,
    Queue,
    Future,
)
from pydantic import Field
from rich import print
from rich.progress import (
    Progress,
    TextColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    BarColumn,
    TextColumn,
)

from analyzer.core.dask_tools import reduceResults
from .condor_tools import setupForCondor
from .executor import Executor
from .preprocessing_tools import preprocess

from analyzer.core.analyzer import runAnalyzerChunks
import analyzer.core.dask_sizes  # noqa

logger = logging.getLogger(__name__)


def constructSampleResult(task, prepped, future_result):
    if future_result is None:
        raise AnalysisRuntimeError(
            f"Could not process any files for sample {task.sample_id}"
        )
    if isinstance(future_result, Exception):
        raise AnalysisRuntimeError(
            f"An exception occurred while processing task {task.sample_id}"
        ) from future_result

    fset, all_results = future_result
    sample_id = task.sample_id
    sample_params = task.sample_params
    processed = prepped.intersect(fset)
    final_result = core_results.SampleResult(
        sample_id=sample_id,
        file_set_ran=prepped,
        file_set_processed=processed,
        params=sample_params,
        results=all_results,
    )
    return final_result


def dumpKeyByteProcessed(result):
    data = core_results.MultiSampleResult.model_validate({result.sample_id: result})
    # data = lz4.frame.compress(pkl.dumps(data.model_dump()))
    return ((result.sample_id, data.toBytes()), result.file_set_processed.events)


def prepWrapper(queue, task, prep_kwargs=None, **kwargs):
    prep_kwargs = prep_kwargs or {}

    client = get_client()
    secede()
    res = preprocess(task, **prep_kwargs)
    merge_futures = runPrepped(client, task, res, **kwargs)
    with dask.annotate(priority=1000):
        results = client.map(
            lambda x: constructSampleResult(task, res, x), merge_futures
        )
        compressed = client.map(dumpKeyByteProcessed, results)
    for f in compressed:
        queue.put(f)
    rejoin()
    return res.nfiles, res.events


class LocalExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def shutdown(self, *args, **kwargs):
        return

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return


def mergeFutures(futures):
    logger.info(f"Merging {len(futures)} futures")
    to_process = [x for x in futures if x is not None and not isinstance(x, Exception)]
    if not to_process:
        return None
    ret = to_process[0]
    for p, r in to_process[1:]:
        iadd(ret[0], p)
        iadd(ret[1], r)
        del r
        gc.collect()

    return ret


def runPrepped(
    client,
    task,
    file_set_prepped,
    timeout=120,
    input_events_per_output=10**7,
    reduction_factor=5,
):
    total_events = file_set_prepped.events
    final_files = math.ceil(total_events / input_events_per_output)

    if task.file_set.form is not None:
        maybe_base_form = ak.forms.from_json(decompress_form(task.file_set.form))
    else:
        maybe_base_form = None

    to_submit = list(file_set_prepped.iterChunks())
    futures = client.map(
        lambda chunk: runAnalyzerChunks(
            task.analyzer,
            chunk,
            task.sample_params,
            known_form=maybe_base_form,
            timeout=timeout,
        ),
        to_submit,
        key=[f"{task.sample_id}-{i}" for i in range(len(to_submit))],
    )
    with dask.annotate(priority=100):
        final_futures = reduceResults(
            client,
            mergeFutures,
            futures,
            reduction_factor=reduction_factor,
            target_final_count=final_files,
            key_suffix="-" + str(task.sample_id),
        )
    return final_futures


class DaskExecutor(Executor):
    max_workers: int
    min_workers: int

    worker_memory: str = "4GB"
    dashboard_address: str | None = "localhost:8789"
    schedd_address: str | None = "localhost:12358"

    adapt: bool = True

    step_size: int | None = 100000

    use_threads: bool = False
    bulk_mode: bool = False
    timeout: int = 360

    reduction_factor: int = 5
    parallel_save: int | None = 8

    def setup(self):
        if self.adapt:
            self._cluster.adapt(
                minimum_jobs=self.min_workers, maximum_jobs=self.max_workers
            )

    def runFutured(self, tasks, result_complete_callback=None):
        client = self._client
        tasks = list(tasks)
        for t in tasks:
            t.file_set.step_size = t.file_set.step_size or self.step_size

        queue = Queue()
        run_kwargs = {
            "reduction_factor": self.reduction_factor,
            "timeout": self.timeout,
        }
        prepped_futures = dict(
            zip(
                client.map(
                    lambda x, kwargs=run_kwargs: prepWrapper(queue, x, **kwargs),
                    tasks,
                    key=[f"prep--{t.sample_id}" for t in tasks],
                ),
                tasks,
            )
        )
        finalizing_futures = set()
        completed = as_completed(prepped_futures)
        start_time = time.time()

        total_files_to_prep = sum(t.file_set.nfiles for t in tasks)
        total_events_to_analyze = 0

        saving_tasks = []
        save_queue = Queue()
        if self.parallel_save is not None:
            save_executor = ProcessPoolExecutor
        else:
            save_executor = LocalExecutor
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            disable=False,
        ) as progress:
            with save_executor(max_workers=self.parallel_save) as pexec:
                prep_bar = progress.add_task(
                    "Pre-Processing", total=total_files_to_prep
                )
                analyze_bar = progress.add_task(
                    "Analyzing", total=total_events_to_analyze
                )
                for batch in completed.batches():
                    for f in batch:
                        try:
                            if f in prepped_futures:
                                i = 0
                                while queue.qsize() > 0:
                                    x = queue.get()
                                    finalizing_futures.add(x)
                                    i += 1
                                    completed.add(x)
                                nfiles, nevents = f.result()
                                total_events_to_analyze += nevents
                                progress.update(prep_bar, advance=nfiles)
                                progress.update(
                                    analyze_bar, total=total_events_to_analyze
                                )
                            elif f in finalizing_futures:
                                res, events = f.result()
                                saving_tasks.append(
                                    pexec.submit(result_complete_callback, *res)
                                )
                                progress.update(analyze_bar, advance=events)
                        except Exception as e:
                            raise e
                            logger.warn(
                                f"An exception occurred while processing a future."
                                f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                                f"{e}"
                            )
                            logger.warn(traceback.format_exc())
                        finally:
                            f.cancel()
                pexec.shutdown(wait=True)

    def run(self, *args, **kwargs):
        self.runFutured(*args, **kwargs)
        return


class LocalDaskExecutor(DaskExecutor):
    executor_type: Literal["dask_local"] = "dask_local"
    processes: bool = True

    def setup(self):
        with open(CONFIG.DASK_CONFIG_PATH) as f:
            defaults = yaml.safe_load(f)
            dask.config.update(dask.config.config, defaults, priority="new")

        log_config_path = Path(CONFIG.CONFIG_PATH) / "worker_logging_config.yaml"
        with open(log_config_path, "rt") as f:
            config = yaml.safe_load(f.read())
            c = base64.b64encode(json.dumps(config).encode("utf-8")).decode("utf-8")
        os.environ["ANALYZER_LOG_CONFIG"] = c

        self._cluster = LocalCluster(
            dashboard_address=self.dashboard_address,
            memory_limit=self.worker_memory,
            n_workers=self.max_workers,
            scheduler_kwargs={"host": self.schedd_address},
            processes=self.processes,
        )

        self._client = Client(self._cluster)


class LPCCondorDask(DaskExecutor):
    executor_type: Literal["dask_condor"] = "dask_condor"
    apptainer_working_dir: str | None = None
    base_working_dir: str | None = None
    venv_path: str | None = None
    x509_path: str | None = None
    base_log_path: str | None = "/uscmst1b_scratch/lpc1/3DayLifetime/"
    temporary_path: str | None = ".temporary"

    extra_files: list[str] = Field(default_factory=list)
    worker_timeout: int | None = 7200
    analysis_root_dir: str | None = "/srv"

    def model_post_init(self, __context):
        self.venv_path = self.venv_path or os.environ.get("VIRTUAL_ENV")
        self.x509_path = self.x509_path or os.environ.get("X509_USER_PROXY")
        self.apptainer_working_dir = self.apptainer_working_dir or os.environ.get(
            "APPTAINER_WORKING_DIR", "."
        )
        self._working_dir = str(Path(".").absolute())

    def setup(self):
        from lpcjobqueue import LPCCondorCluster
        import shutil

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
        with open(log_config_path, "rt") as f:
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
        # import dask_memusage
        # dask_memusage.install(self._cluster.scheduler, "memusage.csv")
        self._client = Client(self._cluster)
        try:
            os.system("ulimit -n 4096")
        except Exception:
            pass
        super().setup()
