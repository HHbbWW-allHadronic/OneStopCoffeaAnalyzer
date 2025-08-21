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
from pydantic import Field, RootModel
from .condor_tools import setupForCondor
from rich import print
from analyzer.configuration import CONFIG

from .executor import Executor

def getPrepCachePath(task):
    base = Path(CONFIG.PREPROCESS_CACHE_PATH)
    target = base / str(task.sample_id) / str()

    

def preprocess(task):
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

    return file_set_prepped
