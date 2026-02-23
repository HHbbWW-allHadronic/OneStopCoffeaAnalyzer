from __future__ import annotations

import fnmatch
import copy
from collections import ChainMap, OrderedDict, namedtuple
from collections.abc import Iterable
from typing import Any


import string


def dotFormat(s, **kwargs):
    parsed = string.Formatter().parse(s)
    s = ""
    for x in parsed:
        s += x[0]
        if x[1] is not None:
            s += str(kwargs[x[1]])
    return s


def dictToDot(dictionary):
    for field, value in dictionary.items():
        if isinstance(value, dict):
            for key, val in dictToDot(value):
                yield f"{field}.{key}", val
        else:
            yield field, value


def flatten(l, limit_to_types=(list,)):
    ret = []
    if isinstance(l, limit_to_types):
        for item in l:
            ret.extend(flatten(item))
    else:
        ret.append(l)
    return ret


def freeze(data):
    if isinstance(data, dict):
        return tuple((x, freeze(y)) for x, y in data.items())
    elif isinstance(data, (list, tuple)):
        return tuple(freeze(x) for x in data)
    else:
        return data


def _mergeUpdate(a: dict[Any, Any], b: dict[Any, Any], current_depth=0, max_depth=None):
    for key in b:
        if key in a:
            if (
                isinstance(a[key], dict)
                and isinstance(b[key], dict)
                and (max_depth is None or current_depth < (max_depth))
            ):
                if current_depth < max_depth:
                    mergeUpdate(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def mergeUpdate(a: dict[Any, Any], b: dict[Any, Any], max_depth=None):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _mergeUpdate(a[key], b[key], current_depth=1, max_depth=max_depth)
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def deepMerge(a: dict[Any, Any], *rest, max_depth=None):
    a = copy.deepcopy(a)
    for d in rest:
        mergeUpdate(a, d, max_depth=max_depth)
    return a


ItemWithMeta = namedtuple("ItemWithMeta", "item metadata")


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

    return ItemWithMeta(current, current_meta)


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
                ret.append(ItemWithMeta(item, item_meta))
            else:
                if isinstance(item, Iterable):
                    ret.extend(globWithMeta(item, rest, item_meta))
    return ret


def deepWalkMeta(directory, pattern=None, current_meta=None, complete_path=None):
    complete_path = complete_path or tuple()
    current_meta = current_meta or ChainMap({})
    for k in directory:
        item = directory[k]
        complete_path = (*complete_path, item.name)
        item_meta = current_meta.new_child(item.metadata).new_child(
            {"name": item.name, "path": complete_path}
        )
        if isinstance(item, Iterable):
            yield from deepWalkMeta(
                item,
                current_meta=item_meta,
                pattern=pattern,
                complete_path=complete_path,
            )
        else:
            if pattern is None or fnmatch.fnmatch(k, pattern):
                yield ItemWithMeta(item, item_meta)


class SimpleCache(OrderedDict):
    "Store items in the order the keys were last added"

    def __init__(self, *args, max_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if self.max_size and (len(self) > self.max_size):
            self.popitem(last=False)

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)


def commonDict(items, key=lambda x: x.metadata):
    i = iter(items)
    ret = copy.deepcopy(dict(key(next(i))))
    for item in i:
        data = key(item)
        for k in data:
            if k in ret and not ret[k] == data[k]:
                del ret[k]
    return ret
