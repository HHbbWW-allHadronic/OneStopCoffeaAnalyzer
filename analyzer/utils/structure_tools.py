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


@ft.singledispatch
def freeze(data):
    return data


@freeze.register
def _(data: dict):
    return frozenset(sorted((freeze(x), freeze(y)) for x, y in data.items()))


@freeze.register
def _(data: list):
    return tuple(freeze(x) for x in data)


@freeze.register
def _(data: list):
    return frozenset(x for x in data)


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


# def doFormatting(s, **kwargs):
#     parsed = string.Formatter().parse(s)
#     s = ""
#     for x in parsed:
#         s += x[0]
#         if x[1] is not None:
#             s += str(kwargs[x[1]])
#     return s
#
# def dictToFrozen(d):
#     return frozenset(sorted(d.items()))
#
#
# def get_git_revision_hash() -> str:
#     return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
#
#
# def get_git_revision_short_hash() -> str:
#     return (
#         subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
#         .decode("ascii")
#         .strip()
#     )
#
#
# def deepMerge(a: dict[Any, Any], b: dict[Any, Any], path=[], overwrite=True):
#     for key in b:
#         if key in a:
#             if isinstance(a[key], dict) and isinstance(b[key], dict):
#                 deepMerge(a[key], b[key], path + [str(key)])
#             else:
#                 a[key] = b[key]
#             # elif a[key] != b[key]:
#             #     raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
#         else:
#             a[key] = b[key]
#     return a
#
#
# def dictIntersect(a, b):
#     if not type(a) == type(b):
#         raise KeyError()
#     if isinstance(a, dict):
#         ret = {}
#         for k in a:
#             if k in b:
#                 d = dictIntersect(a[k], b[k])
#                 isd = isinstance(d, dict)
#                 if (isd and d) or (not isd and d is not None):
#                     ret[k] = d
#         return ret
#     else:
#         if a == b:
#             return a
#
#
# # From https://github.com/CoffeaTeam/coffea/blob/v2023.7.0.rc0/src/coffea/processor/accumulator.py
# T = TypeVar("T")
#
#
# @runtime_checkable
# class Addable(Protocol):
#     def __add__(self: T, other: T) -> T: ...
#
#
# Accumulatable = (
#     Addable | MutableSet["Accumulatable"] | MutableMapping[Any, "Accumulatable"]
# )
#
#
# def add(a: Accumulatable, b: Accumulatable) -> Accumulatable:
#     """Add two accumulatables together, without altering inputs
#
#     This may make copies in certain situations
#     """
#     if isinstance(a, Addable) and isinstance(b, Addable):
#         return operator.add(a, b)
#     if isinstance(a, MutableSet) and isinstance(b, MutableSet):
#         return operator.or_(a, b)
#     elif isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
#         # capture type(X) by shallow copy and clear
#         # since we don't know the signature of type(X).__init__
#         if isinstance(b, type(a)):
#             out = copy.copy(a)
#         elif isinstance(a, type(b)):
#             out = copy.copy(b)
#         else:
#             raise ValueError(
#                 f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
#             )
#         out.clear()
#         lhs, rhs = set(a), set(b)
#         # Keep the order of elements as far as possible
#         for key in a:
#             if key in rhs:
#                 out[key] = add(a[key], b[key])
#             else:
#                 out[key] = (
#                     copy.deepcopy(a[key])
#                     if not isinstance(a[key], DaskMethodsMixin)
#                     else copy.copy(a[key])
#                 )
#         for key in b:
#             if key not in lhs:
#                 out[key] = (
#                     copy.deepcopy(b[key])
#                     if not isinstance(b[key], DaskMethodsMixin)
#                     else copy.copy(b[key])
#                 )
#         return out
#     raise ValueError(
#         f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})"
#     )
#
#
# def iadd(a: Accumulatable, b: Accumulatable) -> Accumulatable:
#     """Add two accumulatables together, assuming the first is mutable"""
#     if isinstance(a, Addable) and isinstance(b, Addable):
#         return operator.iadd(a, b)
#     elif isinstance(a, MutableSet) and isinstance(b, MutableSet):
#         return operator.ior(a, b)
#     elif isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
#         if not isinstance(b, type(a)):
#             raise ValueError(
#                 f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
#             )
#         lhs, rhs = set(a), set(b)
#         # Keep the order of elements as far as possible
#         for key in a:
#             if key in rhs:
#                 a[key] = iadd(a[key], b[key])
#         for key in b:
#             if key not in lhs:
#                 a[key] = (
#                     copy.deepcopy(b[key])
#                     if not isinstance(b[key], DaskMethodsMixin)
#                     else copy.copy(b[key])
#                 )
#         return a
#     raise ValueError(
#         f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})"
#     )
#
#
# def accumulate(
#     items: Iterable[Accumulatable], accum: Accumulatable | None = None
# ) -> Accumulatable | None:
#     gen = (x for x in items if x is not None)
#     try:
#         if accum is None:
#             accum = next(gen)
#             # we want to produce a new object so that the input is not mutated
#             accum = add(accum, next(gen))
#         while True:
#             # subsequent additions can happen in-place, which may be more performant
#             accum = iadd(accum, next(gen))
#     except StopIteration:
#         pass
#     return accum
