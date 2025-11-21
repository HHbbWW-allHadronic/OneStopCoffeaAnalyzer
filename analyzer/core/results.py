from __future__ import annotations
import awkward as ak
from pathlib import Path

from analyzer.core.exceptions import ResultIntegrityError
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
from analyzer.utils.pretty import progbar
from attrs import define, field
import hist
from attrs import define, field


import copy
import abc
import awkward as ak
from typing import Any, Literal, ClassVar
import awkward as ak


import functools as ft

@define
class ResultBase(abc.ABC):
    name: str

    @abc.abstractmethod
    def __iadd__(self, other):
        pass

    @abc.abstractmethod
    def iscale(self, value):
        pass

    @abc.abstractmethod
    def summary(self):
        pass
    
    # @abc.abstractmethod
    # def approxSize(self):
    #     pass

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def scale(self, value):
        ret = copy.deepcopy(self)
        return ret.iscale(value)



@define
class ResultContainer(ResultBase):
    _MAGIC_ID: ClassVar[Literal[b"sstopresult"]] = b"sstopresult"
    _HEADER_SIZE: ClassVar[Literal[4]] = 4

    metadata: dict[str, Any] = field(factory=dict)
    results: dict[str, ResultBase] = field(factory=dict)

    @classmethod
    def peekFile(cls, f):
        maybe_magic = f.read(len(cls._MAGIC_ID))
        if maybe_magic == cls._MAGIC_ID:
            peek_size = int.from_bytes(f.read(cls._HEADER_SIZE), byteorder="big")
            ret = converter.unstructure(pkl.loads(f.read(peek_size)), cls.SummaryClass)
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
                len(cls._MAGIC_ID)
                + cls._HEADER_SIZE : len(cls._MAGIC_ID)
                + cls._HEADER_SIZE
                + peek_size
            ]
            return converter.structure(pkl.loads(peek), cls.SummaryClass)
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
                len(cls._MAGIC_ID)
                + cls._HEADER_SIZE : len(cls._MAGIC_ID)
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

    @define
    class Summary:
        results: dict[str, Any] = field(factory=dict)

    def summary(self):
        return ResultContainer.Summary(
            results={x: y.summary() for x, y in self.results.items()}
        )

    def approxSize(self):
        return sum(x.approxSize() for x in self.results.values()) 

    def addResult(self, res):
        self.results[res.name] = res

    # def __setitem__(self, key, value):
    #     self.results[key] = value

    def __getitem__(self, key):
        return self.results[key]

    def __iter__(self, key):
        return iter(self.results)

    def keys(self):
        return self.results.keys()

    def checkOk(self, other):
        if "_provenance" in self.results:
            if "_provenance" not in other.results:
                raise RuntimeError()
            if (
                not self["_provenance"]
                .file_set.intersect(other["_provenance"].file_set)
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

    def finalize(self, finalizer, converter):
        converter.unstructure(self.results)
        results = finalizer(results)
        self.results = converter.structure(results, dict[str, ResultBase])

        for result in self.results.values():
            result.finalize(finalizer)


@define
class ResultProvenance(ResultBase):
    file_set: FileSet

    def summary(self):
        return self

    def approxSize(self):
        return 30 * len(self.file_set.files)

    def __iadd__(self, other):
        self.file_set += other.file_set
        return self

    def iscale(self, value):
        return self


@define
class Histogram(ResultBase):
    @define
    class Summary:
        axes: Any

    axes: Any
    histogram: hist.Hist

    def summary(self):
        return Histogram.Summary(axes=self.axes)

    def approxSize(self):
        from dask.sizeof import sizeof
        return sizeof(self.histogram.view(flow=True))

    def __iadd__(self, other):
        self.histogram += other.histogram
        return self

    def iscale(self, value):
        self.histogram *= value
        return self


Array = ak.Array | dak.Array | np.ndarray



@define
class ScalableArray(ResultBase):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate(self.array, other.array, axis=0)
        return self

    def iscale(self, value):
        self.array *= value
        return self

    def finalize(self, finalizer, converter):
        self.array = finalizer(self.array)


@define
class RawArray(ResultBase):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate(self.array, other.array, axis=0)
        return self

    def iscale(self, value):
        return self

    def finalize(self, finalizer):
        self.array = finalizer(self.array)


Scalar = dak.Scalar | numbers.Real


@define
class SelectionFlow(ResultBase):
    cuts: list[str]

    cutflow: dict[str, Scalar]
    # n_minus_one: dict[str, Scalar]
    # one_cut: dict[str, Scalar]

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

    def summary(self):
        return self

    def finalize(self, finalizer, converter):
        pass


@define
class RawEventCount(ResultBase):
    count: float

    def __iadd__(self, other):
        self.count += other.count
        return self

    def iscale(self, value):
        return self

    def finalize(self, finalizer, converter):
        pass


@define
class ScaledEventCount(ResultBase):
    count: float

    def __iadd__(self, other):
        self.count += other.count
        return self

    def iscale(self, value):
        self.count *= value
        return self

    def finalize(self, finalizer, converter):
        pass


@define
class RawSelectionFlow(ResultBase):
    cuts: list[str]

    cutflow: dict[str, Scalar]
    n_minus_one: dict[str, Scalar]
    one_cut: dict[str, Scalar]

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

    def finalize(self, finalizer, converter):
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


def loadResults(paths):

    import cProfile
    profiler = cProfile.Profile()
    all_paths = it.chain.from_iterable((Path(".").rglob(x) for x in paths))
    all_paths = paths
    ret = None
    for p in all_paths:
        profiler.enable()
        with open(p, "rb") as f:
            result = ResultContainer.fromBytes(f.read())
        if ret is None:
            ret = result
        else:
            ret += result
        print(result)
        profiler.disable()
    profiler.dump_stats("prof.prof")
    
    return ret
