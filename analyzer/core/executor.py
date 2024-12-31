from rich import print
import copy
import uuid
import os
import coffea.dataset_tools as dst
import abc

import dask
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from pydantic import BaseModel, Field
from typing import Optional, Literal, Union, Annotated, Any
from coffea.util import decompress_form
import awkward as ak

from analyzer.utils.file_tools import compressDirectory
from distributed import LocalCluster

from dataclasses import dataclass
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


@dataclass
class AnalysisTask:
    sample_id: SampleId
    sample_params: SampleParams
    file_set: FileSet
    analyzer: core_analyzer.Analyzer


def giveIdToTasks(tasks: AnalysisTask):
    return {str(uuid.uuid4()): task for task in tasks}


class CondorExecutor(BaseModel):
    executor_type: Literal["condor"] = "condor"
    max_workers: int
    memory: str
    disk: str
    timeout: Optional[int] = None


dict_adapter = TypeAdapter(dict)


class Executor(abc.ABC, BaseModel):
    def setup(self):
        pass

    @abc.abstractmethod
    def run(self, tasks: dict[Any, AnalysisTask]):
        pass


class DaskExecutor(Executor):
    memory: str = "2GB"
    dashboard_address: Optional[str] = "localhost:8787"
    schedd_address: Optional[str] = "localhost:12358"
    max_workers: Optional[int] = 2
    min_workers: Optional[int] = 0
    adapt: bool = True
    step_size: Optional[int] = 100000

    def setup(self):
        if self.adapt and False:
            self._cluster.adapt(
                min_workers=self.min_workers, max_workers=self.max_workers
            )

    def _preprocess(self, tasks):
        to_prep = {
            uid: task.file_set.justUnchunked().toCoffeaDataset()
            for uid, task in tasks.items()
        }
        out, all_items = dst.preprocess(
            to_prep,
            save_form=False,
            skip_bad_files=True,
            uproot_options={"timeout": 1},
            step_size=self.step_size,
        )
        new_filesets = {
            uid: task.file_set.updateFromCoffea(out[uid]).justChunked()
            for uid, task in tasks.items()
        }
        for v in new_filesets.values():
            v.step_size = self.step_size

        return new_filesets

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
            ret[k] = {
                "result": task.analyzer.run(
                    all_events[k][0], task.sample_params
                ).model_dump(),
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
    catch_exceptions: Optional[bool] = False
    step_size: Optional[int] = 100000

    def _preprocess(self, tasks):
        to_prep = {
            uid: task.file_set.justUnchunked().toCoffeaDataset()
            for uid, task in tasks.items()
        }
        out, all_items = dst.preprocess(
            to_prep,
            save_form=False,
            skip_bad_files=True,
            step_size=self.step_size,
            scheduler="single-threaded",
        )
        new_filesets = {
            uid: task.file_set.updateFromCoffea(out[uid]).justChunked()
            for uid, task in tasks.items()
        }
        for v in new_filesets.values():
            v.step_size = self.step_size

        return new_filesets

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
                        ret = accumulate([ret, task.analyzer.run(events, task.sample_params)])
                except Exception as e:
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


class LPCCondorDask(DaskExecutor):
    executor_type: Literal["dask_condor"] = "dask_condor"
    apptainer_working_dir: Optional[str] = None
    venv_path: Optional[str] = None
    x509_path: Optional[str] = None
    base_log_path: Optional[str] = "/uscmst1b_scratch/lpc1/3DayLifetime/"
    temporary_path: Optional[str] = ".temporary"

    extra_files: list[str] = Field(default_factory=list)
    condor_hard_timeout: Optional[int] = 7200
    analysis_root_dir: Optional[str] = "/srv"

    def setup(self):
        """Create a new dask cluster for use with LPC condor."""
        if not LPCQUEUE_AVAILABLE:
            raise NotImplemented("LPC Condor can only be used at the LPC.")
        extra_files = extra_files or []

        apptainer_container = "/".join(
            Path(os.environ["APPTAINER_CONTAINER"]).parts[-2:]
        )

        logpath = Path(self.base_log_path) / os.getlogin() / "dask_logs"
        logpath.mkdir(exist_ok=True, parents=True)

        base = (
            self.apptainer_container
            or Path(os.environ.get("APPTAINER_WORKING_DIR", ".")).resolve()
        )
        venv = self.venv_path or Path(os.environ.get("VIRTUAL_ENV"))
        x509 = self.x509_path or Path(os.environ.get("X509_USER_PROXY")).absolute()

        compressed_env = (
            Path(CONFIG.APPLICATION_DATA) / "compressed" / "environment.tar.gz"
        )
        analyzer_compressed = (
            Path(CONFIG.APPLICATION_DATA) / "compressed" / "analyzer.tar.gz"
        )
        if not compressed_env.exists():
            compressDirectory(
                input_dir=".application_data/venv",
                root_dir=self.analysis_root_dir,
                output=compressed_env,
                archive_type="gztar",
            )
        compressDirectory(
            input_dir=Path(analyzer.__file__).parent.relative_to(
                self.analysis_root_dir
            ),
            root_dir=self.analysis_root_dir,
            output=analyzer_compressed,
            archive_type="gztar",
        )

        transfer_input_files = ["setup.sh", compressed_env, analyzer_compressed]

        if self.extra_files:
            extra_compressed = (
                Path(CONFIG.APPLICATION_DATA) / "compressed" / "extra_files.tar.gz"
            )
            transfer_input_files.append(extra_compressed)
            temp = Path(self.temporary_path)
            extra_files_path = temp / "extra_files/"
            extra_files_path.mkdir(exist_ok=True, parents=True)
            for i in self.extra_files:
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
        # print(s)
        self._cluster = LPCCondorCluster(
            ship_env=False,
            image=apptainer_container,
            memory=memory,
            transfer_input_files=transfer_input_files,
            log_directory=logpath,
            scheduler_options=dict(
                # address=schedd_host,
                dashboard_address=dashboard_address
            ),
            **kwargs,
        )
        self._cluster.adapt(minimum=self.min_workers, maximum=self.max_workers)
        self._client = Client(self._cluster)


AnyExecutor = Annotated[
    Union[LocalDaskExecutor, CondorExecutor, ImmediateExecutor, LPCCondorDask],
    Field(discriminator="executor_type"),
]
