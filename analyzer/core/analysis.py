from __future__ import annotations
import numpy as np
import cProfile, pstats, io

import timeit
import uproot
from superintervals import IntervalMap
import enum
import math
import random

import numbers
import itertools as it
import dask_awkward as dak
import hist
from attrs import asdict, define, make_class, Factory, field
from cattrs import structure, unstructure, Converter
import hist
from coffea.nanoevents import NanoAODSchema
from attrs import asdict, define, make_class, Factory, field
import cattrs
from cattrs import structure, unstructure, Converter
from cattrs.strategies import include_subclasses, configure_tagged_union
import cattrs
from attrs import make_class

from collections.abc import Collection, Iterable
from collections import deque, defaultdict

import contextlib
import uuid
import functools as ft

from rich import print
import copy
import dask
import abc
import awkward as ak
from typing import Any, Literal
from functools import cached_property
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import logging
from rich.logging import RichHandler


@define
class SampleDescription:
    pipelines: list[str]
    collection: str | SourceCollection


@define
class Analysis:
    """
    Complete description of an Analysis
    """

    analyzer: Analyzer
    event_collections: list[CollectionDesc]

    extra_module_paths: list[str] = field(factory=list)
    extra_dataset_paths: list[str] = field(factory=list)
    extra_era_paths: list[str] = field(factory=list)
    extra_executors: dict[str, Executor] = field(factory=dict)




def runAnalysis(analysis):
    default_module_paths = []
    for path in default_module_paths:
        loadRecursive(default_module_paths)
    for path in analysis.extra_module_paths:
        loadRecursive(path)
    
        
    # default_era_paths = []
    # for path in default_era_paths:
    #     loadRecursive(default_module_paths)
    # for path in analysis.extra_era_paths:
    #     loadRecursive(path)
    #     
    # default_dataset_paths = []
    # for path in default_dataset_paths:
    #     loadRecursive(default_module_paths)
    # for path in analysis.extra_dataset_paths:
    #     loadRecursive(path)
    
