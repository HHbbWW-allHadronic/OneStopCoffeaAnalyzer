from __future__ import annotations
import json
import time
import base64

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
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import compress_form, decompress_form
from distributed import Client, LocalCluster, as_completed
from pydantic import Field
from rich import print
from rich.progress import Progress

from .condor_tools import setupForCondor
from .executor import Executor
from .preprocessing_tools import preprocess

from analyzer.core.analyzer import runAnalyzerChunks

logger = logging.getLogger(__name__)


def getSteps(
    files,
    step_size: int | None = None,
    save_form: bool = False,
    uproot_options: dict = {},
):
    awkward.backend(normed_files)
    awkward.typetracer.length_zero_if_typetracer(normed_files)

    for arg in files:
        the_file = uproot.open({arg.file: None}, **uproot_options)
        tree = the_file[arg.object_path]

        num_entries = tree.num_entries

        form_json = None
        form_hash = None
        if save_form:
            form_str = uproot.dask(
                tree,
                ak_add_doc={"__doc__": "title", "typename": "typename"},
                filter_name=no_filter,
                filter_typename=no_filter,
                filter_branch=partial(_is_interpretable, emit_warning=False),
            ).layout.form.to_json()
            # the function cache needs to be popped if present to prevent memory growth
            if hasattr(dask.base, "function_cache"):
                dask.base.function_cache.popitem()

            form_hash = hashlib.md5(form_str.encode("utf-8")).hexdigest()
            form_json = compress_form(form_str)

        target_step_size = num_entries if step_size is None else step_size
        file_uuid = str(the_file.file.uuid)
        n_steps_target = max(round(num_entries / target_step_size), 1)
        actual_step_size = math.ceil(num_entries / n_steps_target)
        out = [
            [
                i * actual_step_size,
                min((i + 1) * actual_step_size, num_entries),
            ]
            for i in range(n_steps_target)
        ]

        out_uuid = file_uuid
        out_steps = out.tolist()

        if out_steps is not None and len(out_steps) == 0:
            out_steps = [[0, 0]]

        array.append(
            {
                "file": arg.file,
                "object_path": arg.object_path,
                "steps": out_steps,
                "num_entries": num_entries,
                "uuid": out_uuid,
                "form": form_json,
                "form_hash_md5": form_hash,
            }
        )

    return array


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


def reduceResults(
    client,
    futures,
    reduction_factor=2,
    target_final_count=10,
    close_to_target_frac=0.7,
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
                mergeFutures,
                futures[i : i + reduction_factor],
                key=f"merge-{layer}-{i}" + str(key_suffix),
            )
            for i in range(0, len(futures), reduction_factor)
        ]
        layer += 1
    return futures


def finalCallback(
    fset,
    all_results,
    sample_id,
    sample_params,
    file_set_prepped,
    result_complete_callback,
):
    processed = file_set_prepped.intersect(fset)
    final_result = core_results.SampleResult(
        sample_id=sample_id,
        file_set_ran=file_set_prepped,
        file_set_processed=processed,
        params=sample_params,
        results=all_results,
    )
    result_complete_callback(final_result.sample_id, final_result)


def runPrepped(
    client,
    task,
    file_set_prepped,
    timeout=120,
    input_events_per_output=10**7,
    reduction_factor=2,
):
    total_events = file_set_prepped.events
    final_files = math.ceil(total_events / input_events_per_output)

    if task.file_set.form is not None:
        maybe_base_form = ak.forms.from_json(decompress_form(task.file_set.form))
    else:
        maybe_base_form = None

    to_submit = list(file_set_prepped.iterChunks())
    with dask.annotate(priority=50):
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
            futures,
            reduction_factor=reduction_factor,
            target_final_count=final_files,
            key_suffix="-" + str(task.sample_id),
        )
    del futures
    return final_futures


def runOneTaskDask(
    task,
    result_complete_callback,
    default_step_size=100000,
    bulk_mode=False,
    scheduler_address=None,
    timeout=120,
    input_events_per_output=10**8,
    reduction_factor=2,
):
    print(f"Starting to process sample {task.sample_id}")
    client = Client(scheduler_address)
    start_time = time.time()
    logger.info(f"Running task {task.sample_id}")
    task.file_set.step_size = task.file_set.step_size or default_step_size

    file_set_prepped = preprocess(task)
    total_events = file_set_prepped.events
    final_files = math.ceil(total_events / input_events_per_output)

    if task.file_set.form is not None:
        maybe_base_form = ak.forms.from_json(decompress_form(task.file_set.form))
    else:
        maybe_base_form = None

    total_events_processed = 0

    if bulk_mode:
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
        with dask.annotate(priority=20):
            final_futures = reduceResults(
                client,
                futures,
                reduction_factor=reduction_factor,
                target_final_count=final_files,
                key_suffix="-" + str(task.sample_id),
            )
        all_results = None
        del futures

        for future in as_completed(final_futures):
            try:
                fset, all_results = future.result()
                processed = file_set_prepped.intersect(fset)
                final_result = core_results.SampleResult(
                    sample_id=task.sample_id,
                    file_set_ran=file_set_prepped,
                    file_set_processed=processed,
                    params=task.sample_params,
                    results=all_results,
                )
                result_complete_callback(final_result.sample_id, final_result)
                total_events_processed += final_result.file_set_processed.events
            except Exception:
                raise

            future.cancel()
            del future
            gc.collect()

    else:
        cds = file_set_prepped.toCoffeaDataset()
        events, report = NanoEventsFactory.from_root(
            cds["files"],
            schemaclass=NanoAODSchema,
            uproot_options=dict(
                allow_read_errors_with_report=True,
            ),
            known_base_form=maybe_base_form,
        ).events()
        r = task.analyzer.run(events, task.sample_params)
        r = r.model_dump()
        ready_to_compute = {"result": r, "report": report}
        computed = dask.compute(ready_to_compute)[0]
        processed = file_set_prepped.justProcessed(computed["report"])
        all_results = computed["result"]

        final_result = core_results.SampleResult(
            sample_id=task.sample_id,
            file_set_ran=file_set_prepped,
            file_set_processed=processed,
            params=task.sample_params,
            results=all_results,
        )
        result_complete_callback(final_result.sample_id, final_result)
        total_events_processed += final_result.file_set_processed.events
    end_time = time.time()
    ellapsed = end_time - start_time
    events_per_sec = total_events_processed / ellapsed
    print(
        f"Finished processing sample {task.sample_id}."
        f"Processed {total_events_processed} events in {ellapsed:0.2f} seconds ({events_per_sec:0.2f} events/s)"
    )


class DaskExecutor(Executor):
    memory: str = "2GB"
    dashboard_address: str | None = "localhost:8789"
    schedd_address: str | None = "localhost:12358"
    max_workers: int | None = 2
    min_workers: int | None = 1
    adapt: bool = True
    step_size: int | None = 100000
    use_threads: bool = False
    parallel_submission: int = 20
    bulk_mode: bool = False
    timeout: int = 120

    def setup(self):
        if self.adapt:
            self._cluster.adapt(
                minimum_jobs=self.min_workers, maximum_jobs=self.max_workers
            )

    def runSerial(self, tasks, result_complete_callback=None):
        for task in tasks:
            try:
                result = runOneTaskDask(
                    task,
                    result_complete_callback,
                    default_step_size=self.step_size,
                    scheduler_address=self._client.cluster.scheduler_address,
                    bulk_mode=self.bulk_mode,
                    timeout=self.timeout,
                )
            except Exception as e:
                logger.warn(
                    f"An exception occurred while processing task {task.sample_id}."
                    f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                    f"{e}"
                )
                logger.warn(traceback.format_exc())

    def runThreaded(self, tasks, result_complete_callback=None):
        with ProcessPoolExecutor(max_workers=self.parallel_submission) as tp:
            futures = {
                tp.submit(
                    runOneTaskDask,
                    v,
                    result_complete_callback,
                    default_step_size=self.step_size,
                    bulk_mode=self.bulk_mode,
                    scheduler_address=self._client.cluster.scheduler_address,
                    timeout=self.timeout,
                    # client=self._client,
                ): v.sample_id
                for v in tasks
            }

            del tasks

            for future in concurrent.futures.as_completed(futures):
                sample_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warn(
                        f"An exception occurred while processing task {sample_id}."
                        f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                        f"{e}"
                    )
                    logger.warn(traceback.format_exc())
                gc.collect()

    def runFutured(self, tasks, result_complete_callback=None):
        client = self._client
        tasks = list(tasks)
        for t in tasks:
            t.file_set.step_size = t.file_set.step_size or self.step_size

        prepped_futures = dict(
            zip(
                client.map(
                    preprocess, tasks, key=[f"prep--{t.sample_id}" for t in tasks]
                ),
                (t for t in tasks),
            )
        )
        finalizing_futures = {}
        completed = as_completed(prepped_futures)
        processed = {task.sample_id: 0 for task in tasks}

        with Progress() as progress:
            task = progress.add_task("Processing", total=None)
            for f in completed:
                try:
                    res = f.result()
                    if isinstance(res, FileSet):
                        task = prepped_futures[f]
                        print(
                            f"Finished preprocessing sample {task.sample_id}. Submitting main job."
                        )
                        final_futures = runPrepped(client, task, res)
                        for new_f in final_futures:
                            finalizing_futures[new_f] = (task, res)
                            completed.add(new_f)
                        del prepped_futures[f]
                        del f

                    else:
                        task, prepped = finalizing_futures[f]
                        print(f"Finished analyzing sample {task.sample_id}.")
                        if res is not None:
                            fset, all_results = res
                            processed[task.sample_id] += fset.events
                            finalCallback(
                                fset,
                                all_results,
                                task.sample_id,
                                task.sample_params,
                                prepped,
                                result_complete_callback,
                            )
                        f.cancel()
                        del finalizing_futures[f]
                        del f
                    gc.collect()
                except Exception as e:
                    logger.warn(
                        f"An exception occurred while processing task {task.sample_id}."
                        f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                        f"{e}"
                    )
                    logger.warn(traceback.format_exc())

    def run(self, *args, **kwargs):
        self.runFutured(*args, **kwargs)
        return
        # if self.use_threads:
        #     # self.runThreaded(*args, **kwargs)
        #     self.runFutured(*args, **kwargs)
        # else:
        #     self.runSerial(*args, **kwargs)


class LocalDaskExecutor(DaskExecutor):
    executor_type: Literal["dask_local"] = "dask_local"

    def setup(self):
        with open(CONFIG.DASK_CONFIG_PATH) as f:
            defaults = yaml.safe_load(f)
            dask.config.update(dask.config.config, defaults, priority="new")
        self._cluster = LocalCluster(
            dashboard_address=self.dashboard_address,
            memory_limit=self.memory,
            n_workers=self.max_workers,
            scheduler_kwargs={"host": self.schedd_address},
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
            # "--preload",
            # "lpcjobqueue.patch",
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
            memory=self.memory,
            transfer_input_files=transfer_input_files,
            log_directory=logpath,
            scheduler_options=dict(dashboard_address=self.dashboard_address),
            job_script_prologue=prologue,
            **kwargs,
        )
        # import dask_memusage
        # dask_memusage.install(self._cluster.scheduler, "memusage.csv")
        self._client = Client(self._cluster)
        super().setup()
