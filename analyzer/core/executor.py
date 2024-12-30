from rich import print
import uuid
import coffea.dataset_tools as dst

import dask
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from pydantic import BaseModel, Field
from typing import Optional
from coffea.util import decompress_form
import awkward as ak

import analyzer.core.results as results
import analyzer.core.analyzer as ac
from dataclasses import dataclass
from analyzer.datasets import FileSet, SampleId, SampleParams
import datetime
import logging
import os
import shutil
import time
from pathlib import Path

import analyzer
from pydantic import TypeAdapter
import dask
import yaml
from analyzer.utils.file_tools import compressDirectory
from distributed import LocalCluster, Client
from rich.progress import Progress

from analyzer.core.results import SubSectorResult, CoreAnalyzerResult, AnalyzerResult
from analyzer.configuration import CONFIG

try:
    from lpcjobqueue import LPCCondorCluster
    from lpcjobqueue.schedd import SCHEDD

    LPCQUEUE_AVAILABLE = True
except ImportError as e:
    LPCQUEUE_AVAILABLE = False


@dataclass
class AnalysisTask:
    sample_id: SampleId
    sample_params: SampleParams
    file_set: FileSet
    analyzer: ac.Analyzer


def giveIdToTasks(tasks):
    return {str(uuid.uuid4()): task for task in tasks}


class CondorExecutor(BaseModel):
    max_workers: int
    memory: str
    disk: str
    timeout: Optional[int] = None


dict_adapter = TypeAdapter(dict)


class DaskExecutor(BaseModel):
    memory: str = "2GB"
    dashboard_address: Optional[str] = "localhost:8787"
    schedd_address: Optional[str] = "localhost:12358"
    max_workers: Optional[int] = 2
    min_workers: Optional[int] = 0

    def model_post_init(self, __context):
        print("Starting scheduler")
        self._cluster = LocalCluster(
            dashboard_address=self.dashboard_address,
            memory_limit=self.memory,
            n_workers=self.max_workers,
            scheduler_kwargs={"host": self.schedd_address},
        )
        self._client = Client(self._cluster)
        print("Done")

    def _preprocess(self, tasks, step_size=50):
        to_prep = {
            uid: task.file_set.justUnchunked().toCoffeaDataset()
            for uid, task in tasks.items()
        }

        print("PREPROCESSING")
        out, all_items = dst.preprocess(
            to_prep,
            save_form=False,
            skip_bad_files=True,
            uproot_options={"timeout": 1},
            step_size=step_size,
        )
        print(out)
        print(all_items)
        print("DONE")
        new_filesets = {
            uid: task.file_set.updateFromCoffea(out[uid]) for uid, task in tasks.items()
        }
        for v in new_filesets.values():
            v.step_size = step_size

        print(new_filesets)
        return new_filesets

    def run(self, tasks):
        tasks = giveIdToTasks(tasks)

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

        print(all_events)

        for k, task in tasks.items():
            ret[k] = {
                "result": task.analyzer.run(
                    all_events[k][0], task.sample_params
                ).model_dump(),
                "report": all_events[k][1],
            }

        print(ret)
        # import code
        # 
        # code.interact(local=locals())
        results = dask.compute(ret)[0]
        print(results)
        for  v in results.values():
            v["result"] = CoreAnalyzerResult(**v["result"])

        print(results)


class LocalDaskExecutor(DaskExecutor):
    def init(self):
        self.cluster = LocalCluster(
            dashboard_address=dashboard_address,
            memory_limit=memory,
            n_workers=max_workers,
            scheduler_kwargs={"host": schedd_address},
        )

    def run(self, units):
        ret = {}
        all_events = {}
        for unit in units:
            cds = unit.file_set.toCoffeaDataset()
            print(cds)
            events, report = NanoEventsFactory.from_root(
                cds["files"],
                schemaclass=NanoAODSchema,
                uproot_options=dict(
                    allow_read_errors_with_report=True,
                    timeout=30,
                ),
                known_base_form=cds["form"],
            ).events()
            all_events[unit.sample_id] = (events, report)
        for unit in units:
            ret[unit.sample_id] = unit.analyzer.run(
                all_events[unit.sample_id][0], unit.sample_params
            )
            print(ret)


class LPCCondorDask(DaskExecutor):
    apptainer_working_dir: str
    venv_path: str
    x509_path: str
    base_log_path: str

    extra_files: list[str] = Field(default_factory=list)
    condor_hard_timeout: int = 7200

    def init(self):
        self.cluster = LocalCluster(
            dashboard_address=dashboard_address,
            memory_limit=memory,
            n_workers=max_workers,
            scheduler_kwargs={"host": schedd_address},
        )

    def run(self, units):
        ret = {}
        all_events = {}
        for unit in units:
            cds = unit.file_set.toCoffeaDataset()
            print(cds)
            events, report = NanoEventsFactory.from_root(
                cds["files"],
                schemaclass=NanoAODSchema,
                uproot_options=dict(
                    allow_read_errors_with_report=True,
                    timeout=30,
                ),
                known_base_form=cds["form"],
            ).events()
            all_events[unit.sample_id] = (events, report)
        for unit in units:
            ret[unit.sample_id] = unit.analyzer.run(
                all_events[unit.sample_id][0], unit.sample_params
            )
            print(ret)


class ImmediateExecutor:
    pass
