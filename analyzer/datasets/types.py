import concurrent.futures
import copy
import enum
import inspect
import itertools as it
import logging
import pickle as pkl
import traceback
import operator as op
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union, Hashable
import functools as ft

import yaml

import awkward as ak
import dask
from analyzer.configuration import CONFIG
from analyzer.utils.file_tools import extractCmsLocation
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import coffea.dataset_tools as cdt
from coffea.util import decompress_form
from pydantic import BaseModel, ConfigDict, Field
import analyzer.datasets.files as adf

import coffea.dataset_tools.apply_processor as cda

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track

@dataclass
class UprootFileSpec:
    object_path: str
    steps: list[list[int]] | list[int] | None


@dataclass
class CoffeaFileSpec(UprootFileSpec):
    steps: list[list[int]]
    num_entries: int
    uuid: str


@dataclass
class CoffeaFileSpecOptional(CoffeaFileSpec):
    steps: list[list[int]] | None
    num_entries: int | None
    uuid: str | None


@dataclass
class DatasetSpec:
    files: dict[str, CoffeaFileSpec]
    metadata: dict[Hashable, Any] | None
    form: str | None


@dataclass
class DatasetSpecOptional(DatasetSpec):
    files: (
        dict[str, str] | list[str] | dict[str, UprootFileSpec | CoffeaFileSpecOptional]
    )


FilesetSpecOptional = dict[str, DatasetSpecOptional]
FilesetSpec = dict[str, DatasetSpec]
