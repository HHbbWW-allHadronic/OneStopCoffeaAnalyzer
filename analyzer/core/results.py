from __future__ import annotations
import awkward as ak
from collections import ChainMap

from rich import print
from pathlib import Path

from analyzer.utils.pretty import progbar
from analyzer.core.exceptions import ResultIntegrityError
from analyzer.utils.file_tools import iterPaths
import numpy as np

import itertools as it

import numbers
import pickle as pkl
import lz4.frame
import dask_awkward as dak
import functools as ft
from cattrs.strategies import include_subclasses, configure_tagged_union
from analyzer.core.event_collection import FileSet
from analyzer.core.serialization import converter
import hist
from analyzer.utils.structure_tools import globWithMeta

from attrs import define, field
import hist
from attrs import define, field


import copy
import abc
import awkward as ak
from typing import Any, Literal, ClassVar
import awkward as ak


import functools as ft


def getArrayMem(array):
    from dask.sizeof import sizeof

    if isinstance(array, ak.highlevel.Array):
        return array.nbytes
    return sizeof(array)


@define
class ResultBase(abc.ABC):
    name: str
    _metadata: dict[str, Any] = field(factory=dict, kw_only=True)

    @property
    def metadata(self):
        return ChainMap(self.getMetadata(), {"type": self.__class__.__name__})

    def getMetadata(self):
        return self._metadata

    @abc.abstractmethod
    def __iadd__(self, other) -> ResultBase:
        pass

    @abc.abstractmethod
    def iscale(self, value) -> ResultBase:
        pass

    @abc.abstractmethod
    def approxSize(self) -> int:
        pass

    @abc.abstractmethod
    def finalize(self) -> ResultBase: ...

    def summary(self):
        return self

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def scale(self, value):
        ret = copy.deepcopy(self)
        return ret.iscale(value)

    def widget(self, *args, **kwargs):
        return None


@define
class ResultGroup(ResultBase):
    _MAGIC_ID: ClassVar[Literal[b"sstopresult"]] = b"sstopresult"
    _HEADER_SIZE: ClassVar[Literal[4]] = 4

    results: dict[str, ResultBase] = field(factory=dict)

    def globWithMeta(self, pattern):
        from analyzer.utils.structure_tools import globWithMeta

        return globWithMeta(self, pattern)

    @classmethod
    def peekFile(cls, f):
        maybe_magic = f.read(len(cls._MAGIC_ID))
        if maybe_magic == cls._MAGIC_ID:
            peek_size = int.from_bytes(f.read(cls._HEADER_SIZE), byteorder="big")
            ret = converter.unstructure(pkl.loads(f.read(peek_size)), ResultGroup)
            return ret
        else:
            return converter.structure(pkl.loads(maybe_magic + f.read())).summary()

    @classmethod
    def peekBytes(cls, data: bytes):
        if data[0 : len(cls._MAGIC_ID)] == cls._MAGIC_ID:
            header_value = data[
                len(cls._MAGIC_ID) : len(cls._MAGIC_ID) + cls._HEADER_SIZE
            ]
            peek_size = int.from_bytes(header_value, byteorder="big")
            peek = data[
                len(cls._MAGIC_ID) + cls._HEADER_SIZE : len(cls._MAGIC_ID)
                + cls._HEADER_SIZE
                + peek_size
            ]
            return converter.structure(pkl.loads(peek), ResultGroup)
        else:
            return converter.structure(pkl.loads(data)).summary()

    @classmethod
    def fromBytes(cls, data: bytes):
        if data[0 : len(cls._MAGIC_ID)] == cls._MAGIC_ID:
            header_value = data[
                len(cls._MAGIC_ID) : len(cls._MAGIC_ID) + cls._HEADER_SIZE
            ]
            peek_size = int.from_bytes(header_value, byteorder="big")
            peek = data[
                len(cls._MAGIC_ID) + cls._HEADER_SIZE : len(cls._MAGIC_ID)
                + cls._HEADER_SIZE
                + peek_size
            ]
            core_data = lz4.frame.decompress(
                data[len(cls._MAGIC_ID) + cls._HEADER_SIZE + peek_size :]
            )
            ret = converter.structure(pkl.loads(core_data), cls)
        else:
            ret = converter.structure(pkl.loads(data), cls)
        return ret

    def toBytes(self, packed_mode=True) -> bytes:
        if packed_mode:
            peek = pkl.dumps(converter.unstructure(self.summary()))
            core_data = lz4.frame.compress(pkl.dumps(converter.unstructure(self)))
            pl = len(peek)
            plb = (pl.bit_length() + 7) // 8
            if plb > self._HEADER_SIZE:
                raise RuntimeError
            return (
                self._MAGIC_ID
                + pl.to_bytes(self._HEADER_SIZE, byteorder="big")
                + peek
                + core_data
            )
        else:
            return pkl.dumps(converter.unstructure(self))

    def summary(self):
        return ResultGroup(
            name=self.name,
            results={x: y.summary() for x, y in self.results.items()},
            metadata=self.metadata,
        )

    def approxSize(self):
        return sum(x.approxSize() for x in self.results.values())

    def addResult(self, res):
        self.results[res.name] = res

    # def __setitem__(self, key, value):
    #     self.results[key] = value

    def __getitem__(self, key):
        return self.results[key]

    def __iter__(self):
        return iter(self.results)

    def keys(self):
        return self.results.keys()

    def checkOk(self, other):
        if "_provenance" in self.results:
            if "_provenance" not in other.results:
                raise RuntimeError()
            if (
                not self["_provenance"]
                .file_set.intersection(other["_provenance"].file_set)
                .empty
            ):
                raise ResultIntegrityError("Overlapping Provenance.")

    def __iadd__(self, other):
        self.checkOk(other)
        for k in other.results:
            if k in self.results:
                self.results[k] += other.results[k]
            else:
                self.addResult(other.results[k])
        return self

    def iscale(self, value):
        for k in self.results:
            self.results[k].iscale(value)
        return self

    def finalize(self, finalizer):
        for result in self.results.values():
            result.finalize(finalizer)


@define
class ResultProvenance(ResultBase):
    file_set: FileSet

    def approxSize(self):
        return 50 * len(self.file_set.files)

    def __iadd__(self, other):
        self.file_set += other.file_set
        return self

    def iscale(self, value):
        return self

    @property
    def chunked_events(self):
        return self.file_set.chunked_events

    def finalize(self, finalizer):
        pass


@define
class Histogram(ResultBase):
    @define
    class Summary(ResultBase):
        axes: Any

        def __iadd__(self, other):
            return self

        def iscale(self, value):
            return self

        def approxSize(self):
            return 0

        def finalize(self, finalizer):
            return self

    axes: Any
    histogram: hist.Hist

    def summary(self):
        return Histogram.Summary(name=self.name, axes=self.axes)

    def approxSize(self):
        from dask.sizeof import sizeof

        return sizeof(self.histogram.view(flow=True))

    def __iadd__(self, other):
        self.histogram += other.histogram
        return self

    def iscale(self, value):
        self.histogram *= value
        return self

    def finalize(self, finalizer):
        return self

    def widget(self, *args, **kwargs):
        from textual_plotext import PlotextPlot

        return None

        widget = PlotextPlot()
        plt = widget.plt
        h = self.histogram
        axes = h.axes
        h = h[{"variation": "central"}]
        if len(h.axes) == 1:
            plt.bar(h.axes[0].centers, h.values())
            return widget
        plt.set_xlabel = axes[0].name
        return None


Array = ak.Array | dak.Array | np.ndarray


@define
class ScalableArray(ResultBase):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate([self.array, other.array], axis=0)
        return self

    def approxSize(self):
        return getArrayMem(self.array)

    def iscale(self, value):
        self.array *= value
        return self

    def finalize(self, finalizer):
        self.array = finalizer(self.array)


@define
class RawArray(ResultBase):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate([self.array, other.array], axis=0)
        return self

    def iscale(self, value):
        return self

    def finalize(self, finalizer):
        self.array = finalizer(self.array)

    def approxSize(self):
        return getArrayMem(array)


@define
class SavedColumns(ResultBase):
    data: dict[str, ak.Array | dak.Array | np.ndarray]

    def __iadd__(self, other):
        if set(self.data) != set(other.data):
            raise RuntimeError()
        for k in self.data:
            self.data[k] = np.concatenate([self.data[k], other.data[k]], axis=0)
        return self

    def iscale(self, value):
        return self

    def finalize(self, finalizer):
        for k in self.data:
            self.data[k] = finalizer(self.data[k])

    def approxSize(self):
        return sum(getArrayMem(x) for x in self.data.values())


Scalar = dak.Scalar | numbers.Real


@define
class SelectionFlow(ResultBase):
    cuts: list[str]

    cutflow: dict[str, Scalar]
    # n_minus_one: dict[str, Scalar]
    # one_cut: dict[str, Scalar]

    def approxSize(self):
        return 30 * len(self.cuts)

    def __iadd__(self, other):
        if self.cuts != other.cuts:
            raise RuntimeError()
        for x in self.cutflow:
            self.cutflow[x] = self.cutflow[x] + other.cutflow[x]
        return self
        # for x in self.n_minus_one:
        #     self.n_minus_one[x] = self.n_minus_one[x] + other.n_minus_one[x]
        # for x in self.one_cut:
        #     self.one_cut[x] = self.one_cut[x] + other.one_cut[x]

    def iscale(self, value):
        for x in self.cutflow:
            self.cutflow[x] = value * self.cutflow[x]
        return self
        # for x in self.n_minus_one:
        #     self.n_minus_one[x] = value * self.n_minus_one[x]
        # for x in self.one_cut:
        #     self.one_cut[x] = value * self.one_cut[x]

    def finalize(self, finalizer):
        pass


@define
class RawEventCount(ResultBase):
    count: float

    def __iadd__(self, other):
        self.count += other.count
        return self

    def approxSize(self):
        return 8

    def iscale(self, value):
        return self

    def finalize(self, finalizer):
        pass


@define
class ScaledEventCount(ResultBase):
    count: float

    def approxSize(self):
        return 8

    def __iadd__(self, other):
        self.count += other.count
        return self

    def iscale(self, value):
        self.count *= value
        return self

    def finalize(self, finalizer):
        pass


@define
class RawSelectionFlow(ResultBase):
    cuts: list[str]

    cutflow: dict[str, Scalar]
    n_minus_one: dict[str, Scalar]
    one_cut: dict[str, Scalar]

    def approxSize(self):
        return 30 * len(self.cuts)

    def __iadd__(self, other):
        if self.cuts != other.cuts:
            raise RuntimeError()
        for x in self.cutflow:
            self.cutflow[x] = self.cutflow[x] + other.cutflow[x]
        for x in self.n_minus_one:
            self.n_minus_one[x] = self.n_minus_one[x] + other.n_minus_one[x]
        for x in self.one_cut:
            self.one_cut[x] = self.one_cut[x] + other.one_cut[x]
        return self

    def iscale(self, value):
        return self

    def finalize(self, finalizer):
        pass


def configureConverter(conv):
    import hist

    @conv.register_structure_hook
    def _(val: Any, _) -> hist.Hist:
        return val

    @conv.register_unstructure_hook
    def _(val: hist.Hist) -> hist.Hist:
        return val

    @conv.register_structure_hook
    def _(val: Scalar, _) -> Scalar:
        return val

    @conv.register_unstructure_hook
    def _(val: Scalar) -> Scalar:
        return val

    @conv.register_structure_hook
    def _(val: Array, _) -> Array:
        return val

    @conv.register_unstructure_hook
    def _(val: Array) -> Array:
        return val

    union_strategy = ft.partial(configure_tagged_union, tag_name="result_type")
    include_subclasses(ResultBase, conv, union_strategy=union_strategy)


configureConverter(converter)


def loadResults(paths, peek_only=False):
    all_paths = paths
    ret = None
    func = ResultGroup.peekBytes if peek_only else ResultGroup.fromBytes
    for p in progbar(iterPaths(all_paths)):
        with open(p, "rb") as f:
            result = func(f.read())
        if ret is None:
            ret = result
        else:
            ret += result
    return ret


def mergeAndScale(results):
    for dataset, meta in globWithMeta(results, ["*"]):
        merged_metadata = copy.deepcopy(meta)
        total = None
        for s in dataset:
            sample_data = dataset[s]
            s_meta = sample_data.metadata
            provenance = sample_data["_provenance"]
            processed_events = provenance.chunked_events
            if s_meta["sample_type"] == "MC":
                lumi = s_meta["era"]["lumi"]
                xs = s_meta["x_sec"]
                sample_data.iscale(lumi * xs / processed_events)
            elif s_meta["sample_type"] == "Data":
                expected_nevents = s_meta["n_events"]
                sample_data.iscale(expected_nevents / processed_events)
            merged_metadata = {
                k: v
                for k, v in merged_metadata.items()
                if k in s_meta and merged_metadata[k] == s_meta[k]
            }
            if total is None:
                total = sample_data
            else:
                total += sample_data
        total.name = dataset.name
        results.addResult(total)
    return results


@define
class ResultStatus:
    dataset_name: str
    sample_name: str
    events_expected: int
    events_found: int

    @property
    def frac_complete(self):
        return self.events_found / self.events_expected


def checkResults(paths):
    results = loadResults(paths, peek_only=True)
    ret = []
    for prov, meta in globWithMeta(results, ["*", "*", "_provenance"]):
        expected = meta["n_events"]
        found = prov.chunked_events
        dataset_name = meta["dataset_name"]
        sample_name = meta["sample_name"]
        ret.append(ResultStatus(dataset_name, sample_name, expected, found))
    return ret
