from __future__ import annotations

import concurrent.futures
import gc
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import yaml

import analyzer.core.results as core_results
import awkward as ak
import coffea.dataset_tools as dst
import dask
from analyzer.configuration import CONFIG
import math
from analyzer.utils.structure_tools import iadd
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from distributed import Client, LocalCluster, as_completed
from pydantic import Field
from rich import print

from .executor import Executor

logger = logging.getLogger(__name__)

def preprocess(tasks, default_step_size=100000, scheduler=None, test_mode=False):
    step_sizes = set(x.file_set.step_size for x in tasks.values())
    step_sizes_not_none = set(
        x.file_set.step_size for x in tasks.values() if x.file_set.step_size is not None
    )
    if len(step_sizes_not_none) > 1:
        raise RuntimeError()

    this_step_size = next(iter(step_sizes))
    to_prep = {uid: task.file_set.justUnchunked() for uid, task in tasks.items()}
    if test_mode:
        scheduler = "single-threaded"
        to_prep = {
            uid: task.file_set.slice(files=slice(0, 1)) for uid, task in tasks.items()
        }

    to_prep = {uid: fs.toCoffeaDataset() for uid, fs in to_prep.items() if not fs.empty}

    if not to_prep:
        return {uid: task.file_set for uid, task in tasks.items()}

    logger.info("Preprocessing %d samples", len(to_prep))
    out, all_items = dst.preprocess(
        to_prep,
        save_form=True,
        skip_bad_files=True,
        step_size=this_step_size or default_step_size,
        scheduler=scheduler,
        allow_empty_datasets=True,
    )
    new_filesets = {
        uid: (
            task.file_set.updateFromCoffea(out[uid]) if uid in out else task.file_set
        ).justChunked()
        for uid, task in tasks.items()
    }

    if test_mode:
        new_filesets = {
            uid: f.slice(chunks=slice(0, 1)) for uid, f in new_filesets.items()
        }
    for v in new_filesets.values():
        v.step_size = v.step_size or default_step_size

    return new_filesets


def mergeFutures(futures):
    futures = [x for x in futures if x is not None]
    if not futures:
        return None
    ret = futures[0]
    for p, r in futures[1:]:
        iadd(ret[0], p)
        iadd(ret[1], r)
    return ret


def reduceResults(client, futures, reduction_count=10, terminal_frac=0.05):
    final_count = math.ceil(terminal_frac * len(futures))
    while len(futures) > final_count:
        futures = [
            client.submit(mergeFutures, futures[i : i + reduction_count])
            for i in range(0, len(futures), reduction_count)
        ]
    return futures


def runOneTaskDask(
    task,
    result_complete_callback,
    default_step_size=100000,
    bulk_mode=False,
    scheduler_address=None,
):
    client = Client(scheduler_address)
    logger.info(f"Running task {task.sample_id}")
    task.file_set.step_size = task.file_set.step_size or default_step_size

    unchunked = task.file_set.justUnchunked()

    if not unchunked.empty:
        to_prep = {task.sample_id: unchunked.toCoffeaDataset()}
        out, all_items = dst.preprocess(
            to_prep,
            save_form=True,
            skip_bad_files=True,
            step_size=task.file_set.step_size,
            allow_empty_datasets=True,
        )
        if out:
            file_set_prepped = task.file_set.updateFromCoffea(
                out[task.sample_id]
            ).justChunked()
        else:
            file_set_prepped = task.file_set.justChunked()
    else:
        file_set_prepped = task.file_set.justChunked()

    if task.file_set.form is not None:
        maybe_base_form = ak.forms.from_json(decompress_form(task.file_set.form))
    else:
        maybe_base_form = None

    if bulk_mode:
        futures = client.map(
            lambda chunk: task.analyzer.runChunks(
                chunk,
                task.sample_params,
                known_form=maybe_base_form,
            ),
            list(file_set_prepped.iterChunks()),
        )
        final_futures = reduceResults(client, futures)
        all_results = None
        for future in as_completed(final_futures):
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
        r = core_results.subsector_adapter.dump_python(r)
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

class DaskExecutor(Executor):
    memory: str = "2GB"
    dashboard_address: str | None = "localhost:8787"
    schedd_address: str | None = "localhost:12358"
    max_workers: int | None = 2
    min_workers: int | None = 1
    adapt: bool = True
    step_size: int | None = 100000
    map_mode: bool = False
    use_threads: bool = False
    parallel_submission: int = 4
    bulk_mode: bool = False

    def setup(self):
        if self.adapt:
            self._cluster.adapt(
                minimum_jobs=self.min_workers, maximum_jobs=self.max_workers
            )

    def _preprocess(self, tasks):
        return preprocess(
            tasks, default_step_size=self.step_size, test_mode=self.test_mode
        )

    def runStandard(self, tasks, result_complete_callback=None):
        ret = {}
        all_events = {}
        file_sets = self._preprocess(tasks)

        for k, t in tasks.items():

            fs = file_sets[k]
            cds = fs.toCoffeaDataset()

            if fs.form is not None:
                maybe_base_form = ak.forms.from_json(decompress_form(fs.form))
            else:
                maybe_base_form = None

            try:
                events, report = NanoEventsFactory.from_root(
                    cds["files"],
                    schemaclass=NanoAODSchema,
                    uproot_options=dict(
                        allow_read_errors_with_report=True,
                    ),
                    known_base_form=maybe_base_form,
                ).events()
                all_events[k] = (events, report)
            except Exception as e:
                logger.warn(
                    f"An exception occurred while preprocessing task {k}.\n"
                    f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                    f"{e}"
                )
        for k, task in tasks.items():
            if k in all_events:
                r = task.analyzer.run(all_events[k][0], task.sample_params)
                r = core_results.subsector_adapter.dump_python(r)
                ret[k] = {
                    "result": r,
                    "report": all_events[k][1],
                }
            else:
                ret[k] = None

        with ThreadPoolExecutor(max_workers=self.parallel_submission) as tp:
            futures = []
            for k, v in ret.items():
                computed = None
                if v is not None:
                    futures.append(
                        tp.submit(lambda x, k=k, v=v: (k, dask.compute(x)), v)
                    )
                else:
                    logger.info(f"Nothing to compute for {k}")
            for future in concurrent.futures.as_completed(futures):
                try:
                    k, computed = future.result()
                    computed = computed[0]
                    if computed is not None:
                        processed = file_sets[k].justProcessed(computed["report"])
                        final_result = core_results.SampleResult(
                            sample_id=tasks[k].sample_id,
                            file_set_ran=file_sets[k],
                            file_set_processed=processed,
                            params=tasks[k].sample_params,
                            results=computed["result"],
                        )
                    else:
                        logger.info(
                            f"Computed result was none for {k}, treating as failure"
                        )
                        final_result = core_results.SampleResult(
                            sample_id=tasks[k].sample_id,
                            file_set_ran=file_sets[k],
                            file_set_processed=file_sets[k].asEmpty(),
                            params=tasks[k].sample_params,
                            results={},
                        )
                    if result_complete_callback is not None:
                        result_complete_callback(k, final_result)

                except Exception as e:
                    raise e
                    logger.warn(
                        f"An exception occurred while processing task {k}."
                        f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                        f"{e}"
                    )

    def runSerial(self, tasks, result_complete_callback=None):
        for task in tasks.values():
            try:
                result = runOneTaskDask(
                    task,
                    result_complete_callback,
                    default_step_size=self.step_size,
                    bulk_mode=self.bulk_mode,
                    client=self._client,
                )
            except Exception as e:
                logger.warn(
                    f"An exception occurred while processing task {task.sample_id}."
                    f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                    f"{e}"
                )
                logger.warn(traceback.format_exc())

    def runThreaded(self, tasks, result_complete_callback=None):

        # from analyzer.utils.debugging import jumpIn
        # jumpIn(**locals(), **globals())
        with ProcessPoolExecutor(max_workers=self.parallel_submission) as tp:
            futures = {
                tp.submit(
                    runOneTaskDask,
                    v,
                    result_complete_callback,
                    default_step_size=self.step_size,
                    bulk_mode=self.bulk_mode,
                    scheduler_address=self._client.cluster.scheduler_address,
                    # client=self._client,
                ): v.sample_id
                for v in tasks.values()
            }
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

    def run(self, *args, **kwargs):
        if self.use_threads:
            self.runThreaded(*args, **kwargs)
        else:
            self.runSerial(*args, **kwargs)
            # self.runStandard(*args, **kwargs)


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
        from lpcjobqueue.schedd import SCHEDD
        import shutil

        with open(CONFIG.DASK_CONFIG_PATH) as f:
            defaults = yaml.safe_load(f)
            dask.config.update(dask.config.config, defaults, priority="new")
        apptainer_container = "/".join(
            Path(os.environ["APPTAINER_CONTAINER"]).parts[-2:]
        )
        Path(os.environ.get("APPTAINER_WORKING_DIR", ".")).resolve()

        logpath = Path(self.base_log_path) / os.getlogin() / "dask_logs"
        logpath.mkdir(exist_ok=True, parents=True)

        logger.info("Deleting old dask logs")
        shutil.rmtree(logpath)
        # for p in self.base_log_path.glob("tmp*"):
        #     shutil.rmtree(p)

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

        logger.info(f"Transfering input files: \n{transfer_input_files}")
        SCHEDD()
        self._cluster = LPCCondorCluster(
            ship_env=False,
            image=apptainer_container,
            memory=self.memory,
            transfer_input_files=transfer_input_files,
            log_directory=logpath,
            scheduler_options=dict(dashboard_address=self.dashboard_address),
            **kwargs,
        )
        print(self._cluster)
        self._client = Client(self._cluster)
        super().setup()
