from __future__ import annotations

import abc
import concurrent.futures
import copy
import gc
import logging
import os
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml

import analyzer
import analyzer.core.analyzer as core_analyzer
import analyzer.core.results as core_results
import awkward as ak
import coffea.dataset_tools as dst
import dask
from analyzer.configuration import CONFIG
from analyzer.datasets import FileSet, SampleId, SampleParams
from analyzer.utils.file_tools import compressDirectory, extractCmsLocation
from analyzer.utils.structure_tools import accumulate
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from distributed import Client, LocalCluster
from pydantic import BaseModel, Field, TypeAdapter
from rich import print

try:
    from lpcjobqueue import LPCCondorCluster
    from lpcjobqueue.schedd import SCHEDD

    LPCQUEUE_AVAILABLE = True
except ImportError as e:
    LPCQUEUE_AVAILABLE = False

NanoAODSchema.warn_missing_crossrefs = False

logger = logging.getLogger(__name__)


class AnalysisTask(BaseModel):
    sample_id: SampleId
    sample_params: SampleParams
    file_set: FileSet
    analyzer: core_analyzer.Analyzer


class PackagedTasks(BaseModel):
    identifier: str
    executor: AnyExecutor
    task: AnalysisTask


def visualize(
    task, file_name, default_step_size=1000000, max_files=1, max_chunks_per_file=1
):
    with open(CONFIG.DASK_CONFIG_PATH) as f:
        defaults = yaml.safe_load(f)
        dask.config.update(dask.config.config, defaults, priority="new")
    task.file_set.step_size = task.file_set.step_size or default_step_size
    to_prep = {
        task.sample_id: task.file_set.slice(files=slice(0, max_files)).toCoffeaDataset()
    }
    out, all_items = dst.preprocess(
        to_prep,
        save_form=True,
        skip_bad_files=True,
        step_size=task.file_set.step_size,
        scheduler="single-threaded",
        allow_empty_datasets=True,
    )
    if out:
        file_set_prepped = task.file_set.updateFromCoffea(
            out[task.sample_id]
        ).justChunked()
    else:
        file_set_prepped = task.file_set.justChunked()

    file_set_prepped = file_set_prepped.slice(chunks=slice(0, max_chunks_per_file))
    cds = file_set_prepped.toCoffeaDataset()
    if task.file_set.form is not None:
        maybe_base_form = ak.forms.from_json(decompress_form(fs.form))
    else:
        maybe_base_form = None
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
    ret = dask.visualize(
        ready_to_compute,
        filename=str(file_name),
        optimize_graph=True,
    )


def runOneTaskDask(task, default_step_size=100000):
    logger.info(f"Running task {task.sample_id}")
    task.file_set.step_size = task.file_set.step_size or default_step_size
    to_prep = {task.sample_id: task.file_set.justUnchunked().toCoffeaDataset()}
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

    cds = file_set_prepped.toCoffeaDataset()
    if task.file_set.form is not None:
        maybe_base_form = ak.forms.from_json(decompress_form(task.file_set.form))
    else:
        maybe_base_form = None
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
    final_result = core_results.SampleResult(
        sample_id=task.sample_id,
        file_set_ran=file_set_prepped,
        file_set_processed=processed,
        params=task.sample_params,
        results=computed["result"],
    )
    return final_result


def giveIdToTasks(tasks: AnalysisTask):
    return {str(uuid.uuid4()): task for task in tasks}


dict_adapter = TypeAdapter(dict)


class Executor(abc.ABC, BaseModel):
    test_mode: bool = False

    def setup(self):
        pass

    @abc.abstractmethod
    def run(self, tasks: dict[Any, AnalysisTask], result_complete_callback=None):
        pass


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
            print(k)
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
                print(k)
                computed = None
                if v is not None:
                    futures.append(tp.submit(lambda x, k=k, v=v: (k, dask.compute(x)), v))
                else:
                    logger.info(f"Nothing to compute for {k}")
            for future in concurrent.futures.as_completed(futures):
                try:
                    k, computed = future.result()
                    print(k)
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

    def runThreaded(self, tasks, result_complete_callback=None):
        with ThreadPoolExecutor(max_workers=self.parallel_submission) as tp:
            futures = [
                tp.submit(runOneTaskDask, v, default_step_size=self.step_size)
                for v in tasks.values()
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    logger.info(f"Successfully got result for {result.sample_id}")
                    result_complete_callback(result.sample_id, result)
                    del result
                except Exception as e:
                    logger.warn(
                        f"An exception occurred while processing task."
                        f"This task will be skipped for the remainder of the analyzer, and the result will need to be patched later:\n"
                        f"{e}"
                    )
                    logger.warn(traceback.format_exc())
                gc.collect()

    def run(self, *args, **kwargs):
        if self.use_threads:
            self.runThreaded(*args, **kwargs)
        else:
            self.runStandard(*args, **kwargs)


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


class ImmediateExecutor(Executor):
    executor_type: Literal["immediate"] = "immediate"
    catch_exceptions: bool = True
    step_size: int = 100000

    def _preprocess(self, tasks):
        return preprocess(
            tasks,
            default_step_size=self.step_size,
            scheduler="single-threaded",
            test_mode=self.test_mode,
        )

    def __run_task(self, k, task):
        fs = self._preprocess({k: task})[k]
        processed = copy.deepcopy(fs)
        cds = fs.toCoffeaDataset()
        ret = None
        for fname, data in cds["files"].items():
            for i, (start, end) in enumerate(data["steps"]):
                try:
                    events = NanoEventsFactory.from_root(
                        {fname: data["object_path"]},
                        schemaclass=NanoAODSchema,
                        delayed=False,
                        entry_start=start,
                        entry_stop=end,
                    ).events()
                    if ret is None:
                        ret = task.analyzer.run(events, task.sample_params)
                    else:
                        ret = accumulate(
                            [ret, task.analyzer.run(events, task.sample_params)]
                        )
                except Exception as e:
                    logger.warn(
                        f"An exception occurred while running {fname} ({start},{end}).\n"
                        f"{e}"
                    )
                    if not self.catch_exceptions:
                        raise
                    else:
                        processed.dropChunk(extractCmsLocation(fname), [start, end])
        return ret, fs, processed

    def run(self, tasks, result_complete_callback=None):

        logger.info(f"Starting run with immediate executor")
        ret = {}

        final_results = {}
        for k, task in tasks.items():
            try:
                logger.info(f"Running task {k}")
                result, fs, processed = self.__run_task(k, task)
                if result is not None:
                    r = core_results.SampleResult(
                        sample_id=tasks[k].sample_id,
                        file_set_ran=fs,
                        file_set_processed=processed,
                        params=tasks[k].sample_params,
                        results=result,
                    )
                    if result_complete_callback is not None:
                        result_complete_callback(k, r)
                    else:
                        final_result[k] = r
            except Exception as e:
                logger.warn(f"An exception occurred while running {k}.\n {e}")
                if not self.catch_exceptions:
                    raise

        return final_results


def setupForCondor(
    analysis_root_dir=None,
    apptainer_dir=None,
    venv_path=None,
    x509_path=None,
    temporary_path=None,
    extra_files=None,
):

    # print(f"{analysis_root_dir = }")
    # print(f"{apptainer_dir = }")
    # print(f"{venv_path = }")
    # print(f"{x509_path = }")
    # print(f"{temporary_path = }")
    extra_files = extra_files or []
    compressed_env = Path(CONFIG.APPLICATION_DATA) / "compressed" / "environment.tar.gz"
    analyzer_compressed = (
        Path(CONFIG.APPLICATION_DATA) / "compressed" / "analyzer.tar.gz"
    )

    if venv_path:
        if not compressed_env.exists():
            compressDirectory(
                input_dir=".application_data/venv",
                root_dir=analysis_root_dir,
                output=compressed_env,
                archive_type="gztar",
            )
    compressDirectory(
        input_dir=Path(analyzer.__file__).parent.relative_to(analysis_root_dir),
        root_dir=analysis_root_dir,
        output=analyzer_compressed,
        archive_type="gztar",
    )

    transfer_input_files = ["setup.sh", compressed_env, analyzer_compressed]

    if extra_files:
        extra_compressed = (
            Path(CONFIG.APPLICATION_DATA) / "compressed" / "extra_files.tar.gz"
        )
        transfer_input_files.append(extra_compressed)
        temp = Path(temporary_path)
        extra_files_path = temp / "extra_files/"
        extra_files_path.mkdir(exist_ok=True, parents=True)
        for i in extra_files:
            src = Path(i)
            shutil.copytree(src, extra_files_path / i)

        compressDirectory(
            input_dir="",
            root_dir=extra_files_path,
            output=extra_compressed,
            archive_type="gztar",
        )
    # if x509_path:
    #     transfer_input_files.append(x509_path)

    return transfer_input_files


class CondorExecutor(Executor):
    executor_type: Literal["condor"] = "condor"
    memory: str = "2GB"
    cpus: int = 1
    disk: str = "2GB"
    files_per_job: int = 1

    apptainer_working_dir: str | None = None
    apptainer_container: str | None = None
    venv_path: str | None = None
    x509_path: str | None = None
    base_log_path: str | None = "/uscmst1b_scratch/lpc1/3DayLifetime/"
    temporary_path: str | None = ".temporary"

    step_size: int = 100000
    timeout: int | None = None

    extra_files: list[str] = Field(default_factory=list)
    output_dir: str | None = None

    executable: str = "condor_exec"

    def splitTask(self, k, task):
        fs = task.file_set
        split = fs.splitFiles(self.files_per_job)
        return {
            f"{k}_{start}_{stop}": PackagedTask(
                identifier=f"{k}_{start}_{stop}",
                executor=ImmediateExecutor(step_size=self.step_size),
                task=AnalysisTask(
                    sample_id=task.sample_id,
                    sample_params=task.sample_params,
                    file_set=f,
                    analyzer=task.analyzer,
                ),
            )
            for (start, stop), f in split.items()
        }

    def makePackagedTasks(self, tasks):
        ret = {}
        for k, task in tasks.items():
            for name, v in self.splitTask(task).items():
                ret[name] = v
        return ret

    def savePackagedTasks(self, packaged):
        ret = []
        condor_path = Path(self.temporary_path) / "condor"
        condor_path.mkdir(exist_ok=True, parents=True)
        for name, task in packaged.items():
            fpath = (condor_path / name).with_extension(".pkl")
            fname = fpath.name
            with open(fpath, "wb") as f:
                pkl.dump(task.model_dump(), f)
            ret.append({"input_path": str(fpath), "input_name": str(fname)})
        return ret

    def run(self, tasks, result_complete_callback=None):
        # import htcondor
        # import classad
        import datetime

        transfer_input_files = setupForCondor(
            # analysis_root_dir=self.apptainer_working_dir,
            venv_path=self.venv_path,
            x509_path=self.x509_path,
            temporary_path=self.temporary_path,
            extra_files=self.extra_files,
        )

        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.f%z")
        log_dir = Path(base_log_path) / now_str
        log_dir.mkdir(exist_ok=True, parents=True)

        task_jobs = self.makePackagedTasks(tasks)
        items = savePackagedTasks(task_jobs)

        job_desc = {
            "executable": self.executable,
            "arguments": f" $(input_file_name) {str(self.output_dir)}",
            "transfer_input_files": "$(input_file)",
            "should_transfer_files": "yes",
            "output": log_dir / "$(input_file_name).out",
            "error": log_dir / "$(input_file_name).err",
            "log": log_dir / "$(input_file_name).log",
            "request_cpus": self.request_cpus,
            "request_memory": self.memory,
            "request_disk": self.disk,
        }

        job = htcondor.Submit(job_desc)
        print(job)
        # schedd = htcondor.Schedd()
        # schedd.submit(items)


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
        with open(CONFIG.DASK_CONFIG_PATH) as f:
            defaults = yaml.safe_load(f)
            dask.config.update(dask.config.config, defaults, priority="new")
        if not LPCQUEUE_AVAILABLE:
            raise NotImplemented("LPC Condor can only be used at the LPC.")
        apptainer_container = "/".join(
            Path(os.environ["APPTAINER_CONTAINER"]).parts[-2:]
        )

        logpath = Path(self.base_log_path) / os.getlogin() / "dask_logs"
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

        logger.info(f"Transfering input files: \n{transfer_input_files}")
        s = SCHEDD()
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


AnyExecutor = Annotated[
    LocalDaskExecutor | CondorExecutor | ImmediateExecutor | LPCCondorDask,
    Field(discriminator="executor_type"),
]
