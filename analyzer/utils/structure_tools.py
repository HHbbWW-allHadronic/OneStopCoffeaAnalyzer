from __future__ import annotations

import fnmatch
import functools as ft
import itertools as it
from collections import ChainMap
from collections.abc import Iterable
from typing import Any

import awkward as ak
import dask_awkward as dak
import numpy as np


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
    current_meta = ChainMap(directory.metadata)
    current = directory
    for k in key:
        current = current[k]
        current_meta = current_meta.new_child(current.metadata).new_child(
            {"name": current.name}
        )

    return current, current_meta


def globWithMeta(directory, pattern, current_meta=None):
    current_meta = current_meta or ChainMap({})
    pattern, *rest = pattern
    ret = []
    for k in directory:
        if fnmatch.fnmatch(k, pattern):
            item = directory[k]
            item_meta = current_meta.new_child(item.metadata).new_child(
                {"name": item.name}
            )
            if not rest:
                ret.append((item, item_meta))
            else:
                if isinstance(item, Iterable):
                    ret.extend(globWithMeta(item, rest, item_meta))
    return ret
