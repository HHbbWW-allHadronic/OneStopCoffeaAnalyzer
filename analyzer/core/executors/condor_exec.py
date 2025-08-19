from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal


from pydantic import Field

# from analyzer.core.executors.tasks import PackagedTask, AnalysisTask
from analyzer.core.executors.executor import Executor

logger = logging.getLogger(__name__)


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
        import analyzer.core.executors.tasks as exec_tasks
        fs = task.file_set
        split = fs.splitFiles(self.files_per_job)
        return {
            f"{k}_{start}_{stop}": exec_tasks.PackagedTask(
                identifier=f"{k}_{start}_{stop}",
                executor=ImmediateExecutor(step_size=self.step_size),
                task=exec_tasks.AnalysisTask(
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
        savePackagedTasks(task_jobs)

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

        htcondor.Submit(job_desc)
        # schedd = htcondor.Schedd()
        # schedd.submit(items)
