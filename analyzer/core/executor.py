from __future__ import annotations
from rich import print
import copy
import uuid
import os
import coffea.dataset_tools as dst
import abc

import dask
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Any, ClassVar
from coffea.util import decompress_form
import awkward as ak

from analyzer.utils.file_tools import compressDirectory
from distributed import LocalCluster

import os
import shutil
from pathlib import Path

from pydantic import TypeAdapter
import dask
from analyzer.utils.file_tools import compressDirectory
from analyzer.utils.structure_tools import accumulate
from distributed import LocalCluster, Client

from analyzer.configuration import CONFIG
from analyzer.datasets import FileSet, SampleId, SampleParams

import analyzer.core.analyzer as core_analyzer
import analyzer.core.results as core_results

try:
    from lpcjobqueue import LPCCondorCluster
    from lpcjobqueue.schedd import SCHEDD

    LPCQUEUE_AVAILABLE = True
except ImportError as e:
    LPCQUEUE_AVAILABLE = False


EXECUTORS = {}


class AnalysisTask(BaseModel):
    sample_id: SampleId
    sample_params: SampleParams
    file_set: FileSet
    analyzer: core_analyzer.Analyzer


class PackagedTask(BaseModel):
    identifier: str
    executor: AnyExecutor
    task: AnalysisTask


def giveIdToTasks(tasks: AnalysisTask):
    return {str(uuid.uuid4()): task for task in tasks}


dict_adapter = TypeAdapter(dict)


class Executor(abc.ABC, BaseModel):

    def setup(self):
        pass

    @abc.abstractmethod
    def run(self, tasks: dict[Any, AnalysisTask]):
        pass


def preprocess(tasks, default_step_size=100000, scheduler=None):
    step_sizes = set(x.file_set.step_size for x in tasks.values())
    if len(step_sizes) != 1:
        raise RuntimeError()

    this_step_size = next(iter(step_sizes))
    to_prep = {
        uid: task.file_set.justUnchunked().toCoffeaDataset()
        for uid, task in tasks.items()
        if not task.file_set.justUnchunked().empty
    }
    # for z in to_prep.values():
    #     for i, x in list(enumerate(z["files"])):
    #         if i % 2 == 1:
    #             del z["files"][x]
    if not to_prep:
        return {uid: task.file_set for uid, task in tasks.items()}

    out, all_items = dst.preprocess(
        to_prep,
        save_form=False,
        skip_bad_files=True,
        step_size=this_step_size or default_step_size,
        scheduler=scheduler,
    )
    new_filesets = {
        uid: task.file_set.updateFromCoffea(out[uid]).justChunked()
        for uid, task in tasks.items()
    }
    for v in new_filesets.values():
        v.step_size = default_step_size

    return new_filesets


class DaskExecutor(Executor):
    memory: str = "2GB"
    dashboard_address: str | None = "localhost:8787"
    schedd_address: str | None = "localhost:12358"
    max_workers: int | None = 2
    min_workers: int | None = 0
    adapt: bool = True
    step_size: int | None = 100000

    def setup(self):
        if self.adapt and False:
            self._cluster.adapt(
                min_workers=self.min_workers, max_workers=self.max_workers
            )

    def _preprocess(self, tasks):
        return preprocess(tasks, default_step_size=self.step_size)

    def run(self, tasks):
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

            events, report = NanoEventsFactory.from_root(
                cds["files"],
                schemaclass=NanoAODSchema,
                uproot_options=dict(
                    allow_read_errors_with_report=True,
                    timeout=30,
                ),
                known_base_form=maybe_base_form,
            ).events()
            all_events[k] = (events, report)
        for k, task in tasks.items():
            r = task.analyzer.run(all_events[k][0], task.sample_params)
            r = core_results.subsector_adapter.dump_python(r)
            ret[k] = {
                "result": r,
                "report": all_events[k][1],
            }
        results = dask.compute(ret)[0]

        final_result = {}
        for k, v in results.items():
            processed = file_sets[k].justProcessed(v["report"])
            final_result[k] = core_results.SampleResult(
                sample_id=tasks[k].sample_id,
                file_set_ran=file_sets[k],
                file_set_processed=processed,
                params=tasks[k].sample_params,
                results=v["result"],
            )

        return final_result


class LocalDaskExecutor(DaskExecutor):
    executor_type: Literal["dask_local"] = "dask_local"

    def setup(self):
        self._cluster = LocalCluster(
            dashboard_address=self.dashboard_address,
            memory_limit=self.memory,
            n_workers=self.max_workers,
            scheduler_kwargs={"host": self.schedd_address},
        )
        self._client = Client(self._cluster)
        super().setup()


class ImmediateExecutor(Executor):
    executor_type: Literal["immediate"] = "immediate"
    catch_exceptions: bool | None = True
    step_size: int | None = 100000

    def _preprocess(self, tasks):
        return preprocess(
            tasks, default_step_size=self.step_size, scheduler="single-threaded"
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
                    # if i % 2 == 1:
                    #     raise RuntimeError()
                except Exception as e:
                    print(e)
                    if not self.catch_exceptions:
                        raise
                    processed.dropChunk(fname, [start, end])
        return ret, fs, processed

    def run(self, tasks):
        ret = {}
        for k, task in tasks.items():
            ret[k] = self.__run_task(k, task)

        final_result = {}
        for k, (result, fs, processed) in ret.items():
            final_result[k] = core_results.SampleResult(
                sample_id=tasks[k].sample_id,
                file_set_ran=fs,
                file_set_processed=processed,
                params=tasks[k].sample_params,
                results=result,
            )

        print(final_result)

        return final_result


def setupForCondor(
    analysis_root_dir=None,
    venv_path=None,
    x509_path=None,
    temporary_path=None,
    extra_files=None,
):
    extra_files = extra_files or []
    base = (
        analysis_root_dir
        or Path(os.environ.get("APPTAINER_WORKING_DIR", ".")).resolve()
    )
    venv = venv_path or Path(os.environ.get("VIRTUAL_ENV"))
    x509 = x509_path or Path(os.environ.get("X509_USER_PROXY")).absolute()

    compressed_env = Path(CONFIG.APPLICATION_DATA) / "compressed" / "environment.tar.gz"
    analyzer_compressed = (
        Path(CONFIG.APPLICATION_DATA) / "compressed" / "analyzer.tar.gz"
    )
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
    if x509:
        transfer_input_files.append(x509)

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

    def run(self, tasks):
        # import htcondor
        # import classad
        import datetime

        transfer_input_files = setupForCondor(
            analysis_root_dir=self.apptainer_working_dir,
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
            "arguments": f"$(input_file_name) {str(self.output_dir)}",
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
    venv_path: str | None = None
    x509_path: str | None = None
    base_log_path: str | None = "/uscmst1b_scratch/lpc1/3DayLifetime/"
    temporary_path: str | None = ".temporary"

    extra_files: list[str] = Field(default_factory=list)
    condor_hard_timeout: int | None = 7200
    analysis_root_dir: str | None = "/srv"

    def setup(self):
        if not LPCQUEUE_AVAILABLE:
            raise NotImplemented("LPC Condor can only be used at the LPC.")
        apptainer_container = "/".join(
            Path(os.environ["APPTAINER_CONTAINER"]).parts[-2:]
        )

        logpath = Path(self.base_log_path) / os.getlogin() / "dask_logs"
        logpath.mkdir(exist_ok=True, parents=True)

        transfer_input_files = setupForCondor(
            apptainer_working_dir=self.apptainer_working_dir,
            venv_path=self.venv_path,
            x509_path=self.x509_path,
            temporary_path=self.temporary_path,
            extra_file=self.extra_files,
        )
        kwargs = {}
        kwargs["worker_extra_args"] = [
            *dask.config.get("jobqueue.lpccondor.worker_extra_args"),
            # "--preload",
            # "lpcjobqueue.patch",
        ]
        kwargs["job_extra_directives"] = {"+MaxRuntime": timeout}
        kwargs["python"] = f"{venv}/bin/python"

        logger.info(f"Transfering input files: \n{transfer_input_files}")
        s = SCHEDD()
        self._cluster = LPCCondorCluster(
            ship_env=False,
            image=apptainer_container,
            memory=memory,
            transfer_input_files=transfer_input_files,
            log_directory=self.base_log_path,
            scheduler_options=dict(dashboard_address=dashboard_address),
            **kwargs,
        )
        self._client = Client(self._cluster)
        super().setup()


AnyExecutor = Annotated[
    LocalDaskExecutor | CondorExecutor | ImmediateExecutor | LPCCondorDask,
    Field(discriminator="executor_type"),
]
