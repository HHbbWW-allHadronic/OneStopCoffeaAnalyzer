from __future__ import annotations

import copy
import operator
import subprocess
import string
from collections.abc import MutableMapping, MutableSet
from typing import Iterable, TypeVar, Any
import numpy as np
import cProfile, pstats, io

import timeit
import uproot
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


def freeze(data):
    if isinstance(data, dict):
        return frozenset((freeze(x), freeze(y)) for x, y in data.items())
    elif isinstance(data, list):
        return frozenset(freeze(x) for x in data)
    else:
        return data


def mergeUpdate(a: dict[Any, Any], b: dict[Any, Any]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                mergeUpdate(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def getWithMeta(directory, key):
    if isinstance(key, str):
        key = key.split(".")
    current_meta = copy.deepcopy(directory.metadata)
    current = directory
    for k in key:
        current = current[k]
        current_meta.update(current.metadata)

    return current, current_meta
