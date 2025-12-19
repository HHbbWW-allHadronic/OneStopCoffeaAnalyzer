from __future__ import annotations
import numpy as np

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

# Define the log message format
FORMAT = "%(message)s"

# Configure basic logging with RichHandler
logging.basicConfig(
    level="WARNING",  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    format=FORMAT,
    handlers=[RichHandler()],
)

# Get a logger instance
log = logging.getLogger("my_app")


logger = logging.getLogger("analyzer")


EVENTS = "EVENTS"


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


class EventBackend(str, enum.Enum):
    coffea_virtual = "coffea_virtual"
    coffea_dask = "coffea_dask"
    coffea_imm = "coffea_eager"
    rdf = "rdf"


class DataType(str, enum.Enum):
    mc = "mc"
    data = "data"


def register_module(input_columns, output_columns, configuration=None, params=None):
    configuration = configuration or {}
    params = params or {}

    def wrapper(func):
        getParameterSpec = lambda x: ModuleParameterSpec(params)
        run = func
        if callable(input_columns):
            inputs = input_columns
        else:
            inputs = lambda self, metadata: [Column(x) for x in input_columns]
        if callable(output_columns):
            outputs = output_columns
        else:
            output = lambda self, metadata: [Column(x) for x in output_columns]
        cls = make_class(
            func.__name__,
            configuration,
            bases=(AnalyzerModule,),
            class_body=dict(
                getParameterSpec=getParameterSpec,
                run=run,
                inputs=inputs,
                outputs=outputs,
            ),
        )
        return cls

    return wrapper


@define
class SourceCollection(abc.ABC):
    nevents: int

    @abc.abstractmethod
    def getSources(self, sources: Iterable[EventSource]) -> SourceCollection: ...

    @abc.abstractmethod
    def preprocess(self): ...


@frozen
class LocatedPath:
    complete_path: str
    location: str


class DasCollection(SourceCollection):
    das_path: str
    tree_name: str = "Events"
    schema_name: str | None = None

    _files: set[frozenset[LocatedPath]] | None = None

    def preprocess(self):
        self._files = getFilesDas(self.das_path)

    def getSources(self):
        if self._files is None:
            self.preprocess()
        return [
            FileSource(
                file_path=next(iter(x)),
                tree_name=self.tree_name,
                schema_name=self.schema_name,
            )
            for x in self._files
        ]


@define
class EventSource(abc.ABC):
    @abc.abstractmethod
    def loadEvents(
        self, backend, metadata, event_start=None, event_stop=None, view_kwargs=None
    ) -> ColumnView: ...

    @abc.abstractmethod
    def chunks(self, chunk_size, max_chunks=None) -> Iterable[SourceChunk]: ...


def getFileEvents(file_path, tree_name):
    tree = uproot.open({file_path: None}, **kwargs)[tree_name]
    nevents = tree.num_entries
    return nevents


class FileSource(EventSource):
    file_path: str
    tree_name: str = "Events"
    schema_name: str | None = None

    _nevents: int | None = field(default=None, eq=False)

    def getNumEvents(self, **kwargs):
        if self._nevents is not none:
            return self._nevents
        self._nevents = getFileEvents(self.file_path, self.tree_name)
        return nevents

    def chunks(self, chunk_size, max_chunks=None, exec_function=None, **kwargs):
        nevents = self.getNumEvents(**kwargs)
        nchunks = max(round(nevents / chunk_size), 1)
        chunk_size = math.ceil(nevents / nchunks)
        chunks = [
            [
                i * chunk_size,
                min((i + 1) * chunk_size, nevents),
            ]
            for i in range(nchunks)
        ]
        for start, stop in chunks:
            yield ChunkedRootFile(source_file=self, event_start=start, event_stop=stop)

    def loadEvents(self, backend, start=None, stop=None, view_kwargs=None):
        view_kwargs = view_kwargs or {}
        view_kwargs["backend"] = backend
        if backend == "coffea-virtual":
            events = NanoEventsFactory.from_root(
                {self.source_file.file_path: self.source_file.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="virtual",
            ).events()
        elif backend == "coffea-dask":
            events = NanoEventsFactory.from_root(
                {self.source_file.file_path: self.source_file.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="dask",
            ).events()
        elif backend == "coffea-eager":
            events = NanoEventsFactory.from_root(
                {self.source_file.file_path: self.source_file.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="eager",
            ).events()
        elif backend == "RDF":
            pass

        return ColumnView.fromEvents(events, **view_kwargs)


class Sample:
    name: str
    x_sec: float | None = None
    event_source: SourceCollection


class Dataset:
    name: str
    era: str | Any
    data_type: DataType
    samples: list[Sample] = field(factory=list)


class SourceChunk(EventSource):
    source: EventSource
    event_start: int
    event_stop: int

    @property
    def metadata(self):
        return self.source.metadata

    def chunks(self, chunk_size, max_chunks=None, **kwargs):
        nevents = end - start
        nchunks = max(round(nevents / chunk_size), 1)
        chunk_size = math.ceil(nevents / nchunks)
        chunks = [
            [
                i * chunk_size,
                min((i + 1) * chunk_size, nevents),
            ]
            for i in range(nchunks)
        ]
        for start, stop in chunks:
            yield ChunkedRootFile(source_file=self, event_start=start, event_stop=stop)

    def loadEvents(self, backend, view_kwargs=None):
        return self.source_file.loadEvents(
            backend,
            start=self.event_start,
            stop=self.event_stop,
            view_kwargs=view_kwargs,
        )

    def overlaps(self, other):
        same_file = self.source_file.file_id == other.source_file.file_id
        if not same_file:
            return False
        return (
            self.event_start <= other.event_stop
            and other.event_start <= self.event_stop
        )


@define
class EventMultiSet(EventCollection):
    events_collections: list[EventCollection]

    def iterChunks(self, chunk_size):
        pass


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


def coerceFields(data):
    if isinstance(data, str):
        return tuple(data.split("."))
    elif isinstance(data, Column):
        return data.fields
    else:
        return data


@define(frozen=True)
class Column:
    fields: tuple[str, ...] = field(converter=coerceFields)

    @property
    def parent(self):
        return self[:-1]

    def parts(self):
        return tuple(self.path)

    def contains(self, other):
        if len(self) > len(other):
            return False
        return other[: len(self)] == self

    def commonParent(self, other):
        l = []
        for x, y in zip(self, other):
            if x == y:
                l.append(x)
            else:
                break
        ret = Column(tuple(l))
        return ret

    def extract(self, events):
        return ft.reduce(lambda x, y: x[y], self.fields, events)

    def __eq__(self, other):
        return self.fields == other.fields

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Column(self.fields.__getitem__(key))
        else:
            return self.fields[key]

    def __add__(self, other):
        return Column(self.fields + Column(other).fields)

    def __radd__(self, other):
        return Column(Column(other).fields + self.fields)

    def __iter__(self):
        return iter(self.fields)

    def __hash__(self):
        return hash(self.fields)


def daskFinalizer(data):
    if isinstance(data, list | dict):
        data = dask.compute(data)[0]
    elif isinstance(data, ak.Array):
        data = data.to_numpy()
    return data


def daskFinalizer(data):
    if isinstance(data, list | dict):
        data = dask.compute(data)[0]
    elif isinstance(data, ak.Array):
        data = data.to_numpy()
    return data


@define
class AnalyzerResult(abc.ABC):
    name: str
    metadata: dict[str, Any] = field(factory=dict)

    @abc.abstractmethod
    def __iadd__(self, other):
        pass

    @abc.abstractmethod
    def iscale(self, value):
        pass

    # @abc.abstractmethod
    # def summary(self):
    #     pass
    #
    # @abc.abstractmethod
    # def getApproxSize(self):
    #     pass

    def finalize(self, finalizer):
        return

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def scale(self, value):
        ret = copy.deepcopy(self)
        return ret.iscale(value)


@define
class ExecutionResult(AnalyzerResult):
    _MAGIC_ID: ClassVar[Literal[b"sstopresult"]] = b"sstopresult"
    _HEADER_SIZE: ClassVar[Literal[4]] = 4

    results: ResultContainer = field(factory=ResultContainer)

    result_provenance: ResultProvenance

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
            core_data = data[len(cls._MAGIC_ID) + cls._HEADER_SIZE + peek_size :]
            ret = converter.structure(pkl.loads(core_data), cls)
        else:
            ret = converter.structure(pkl.loads(data), cls)
        return ret

    def toBytes(self, packed_mode=True) -> bytes:
        if packed_mode:
            peek = pkl.dumps(converter.unstructure(self.summary()))
            core_data = pkl.dumps(converter.unstructure(self))
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
class ResultGroup(AnalyzerResult):
    results: dict[str, AnalyzerResult] = field(factory=dict)

    def __iadd__(self, other):
        for k in self.results:
            self.results[k] += other.results[k]

    def addResult(self, result):
        if result.name in self.results:
            raise KeyError(
                f"Result container already contains a result with name {result.name}"
            )
        self.results[result.name] = result

    def summary(self):
        return


    def iscale(self, value):
        for k in self.results:
            self.results[k].iscale(value)

    def finalize(self, finalizer):
        uns = unstructure(self.results)
        results = finalizer(results)
        self.results = structure(results, dict[str, AnalyzerResult])

        for result in self.results.values():
            result.finalize(finalizer)


class Executor(abc.ABC):
    @abc.abstractmethod
    def run(self, analyzer, tasks: list[AnalysisTask], result_complete_callback=None):
        pass

    def setup(self):
        pass

    def teardown(self):
        pass

    def __exit__(self, type, value, traceback):
        self.teardown()

    def __enter__(self):
        self.setup()


Pipeline = list[AnalyzerModule]


@define
class CollectionDesc:
    pipelines: list[str]
    collection: EventCollection


@define
class Analysis:
    """
    Complete description of an Analysis
    """

    analyzer: Analyzer
    event_collections: list[CollectionDesc]

    ignore_builtin_modules: bool = False
    ignore_builtin_datasets: bool = False
    extra_module_paths: list[str] = field(factory=list)
    extra_source_paths: list[str] = field(factory=list)
    extra_excutors: dict[str, Executor] = field(factory=dict)


@define
class Histogram(AnalyzerResult):
    histogram: hist.Hist

    def __iadd__(self, other):
        self.hist += other.hist

    def iscale(self, value):
        self.hist *= value


@define
class ScalableArray(AnalyzerResult):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate(self.array, other.array, axis=0)

    def iscale(self, value):
        self.array *= value

    def finalize(self, finalizer):
        self.array = finalizer(self.array)


@define
class RawArray(AnalyzerResult):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate(self.array, other.array, axis=0)

    def iscale(self, value):
        pass

    def finalize(self, finalizer):
        self.array = finalizer(self.array)


Scalar = dak.Scalar | numbers.Real


@define
class SelectionFlow(AnalyzerResult):
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

    def iscale(self, value):
        for x in self.cutflow:
            self.cutflow[x] = value * self.cutflow[x]
        for x in self.n_minus_one:
            self.n_minus_one[x] = value * self.n_minus_one[x]
        for x in self.one_cut:
            self.one_cut[x] = value * self.one_cut[x]

    def finalize(self, finalizer):
        pass


@define
class RawEventCount(AnalyzerResult):
    count: float

    def __iadd__(self, other):
        self.count += other.count

    def iscale(self, value):
        pass

    def finalize(self, finalizer):
        pass


@define
class ScaledEventCount(AnalyzerResult):
    count: float

    def __iadd__(self, other):
        self.count += other.count

    def iscale(self, value):
        self.count *= value

    def finalize(self, finalizer):
        pass


@define
class RawSelectionFlow(AnalyzerResult):
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

    def iscale(self, value):
        return

    def finalize(self, finalizer):
        pass


@define
class ColumnCollection:
    columns: set[Column]

    def __iter__(self):
        return iter(columns)

    def contains(self, other: Column):
        return any(x.contains(other) for x in self.columns)

    def intersect(self, other: ColumnCollection):
        ret = {
            x
            for x in self.columns
            if any((x.contains(o) or o.contains(x)) for o in other)
        }


def getAllColumns(events, cur_col=None):
    if fields := getattr(events, "fields"):
        ret = set()
        for f in fields:
            if cur_col is not None:
                n = cur_col + f
            else:
                n = Column(f)
            ret |= getAllColumns(events[f], n)

        if cur_col is not None:
            ret.add(cur_col)
        return ret
    else:
        return {cur_col}


NanoAODSchema.warn_missing_crossrefs = False


@define(frozen=True)
class ModuleParameterValues:
    param_values: frozenset[tuple[str, Any]]
    spec: ModuleParameterSpec | None = field(eq=False, default=None)

    def withNewValues(self, values):
        d = dict(self.param_values)
        d.update(values)
        return self.spec.getWithValues(d)

    def asdict(self):
        return dict(self.param_values)

    def __hash__(self):
        return hash(self.param_values)

    def __getitem__(self, key):
        return self.asdict()[key]

    def __rich_repr__(self):
        yield "param_values", self.asdict()

    def __repr__(self):
        return f"ModuleParameterValues({self.asdict()})"

    def __str__(self):
        return f"ModuleParameterValues({self.asdict()})"


@define(frozen=True)
class PipelineParameterValues:
    values: frozenset[tuple[str, ModuleParameterValues]]
    spec: PipelineParameterSpec

    def __hash__(self):
        return hash(self.values)

    def asdict(self):
        return {x: y.asdict() for x, y in self.values}

    def __getitem__(self, key):
        found = next(x[1] for x in self.values if x[0] == key)
        return ModuleParameterValues(found.param_values, self)

    def withNewValues(self, new_data):
        d = self.asdict()
        mergeUpdate(d, new_data)
        return self.spec.getWithValues(d)

    def __rich_repr__(self):
        yield "values", self.values

    def __str__(self):
        return f"values: {self.values}"

    def __repr__(self):
        return str(self)


# def mergeAnalyzerValues(first, *rest):
#     while rest:
#
# def moduleValues(first, *rest):
#     while rest:


@define(frozen=True)
class ModuleParameterValues:
    param_values: frozenset[tuple[str, Any]]
    spec: ModuleParameterSpec | None = field(eq=False, default=None)

    def withNewValues(self, values):
        d = dict(self.param_values)
        d.update(values)
        return self.spec.getWithValues(d)

    def __iter__(self):
        return iter(self.param_values)

    def asdict(self):
        return dict(self.param_values)

    def __hash__(self):
        return hash(self.param_values)

    def __getitem__(self, key):
        return self.asdict()[key]

    def __rich_repr__(self):
        yield "param_values", self.asdict()

    def __repr__(self):
        return f"ModuleParameterValues({self.asdict()})"

    def __str__(self):
        return f"ModuleParameterValues({self.asdict()})"


@define(frozen=True)
class PipelineParameterValues:
    values: frozenset[tuple[str, ModuleParameterValues]]
    spec: PipelineParameterSpec

    def __hash__(self):
        return hash(self.values)

    def asdict(self):
        return {x: y.asdict() for x, y in self.values}

    def __getitem__(self, key):
        found = next(x[1] for x in self.values if x[0] == key)
        return ModuleParameterValues(found.param_values, self)

    def withNewValues(self, new_data):
        d = self.asdict()
        mergeUpdate(d, new_data)
        return self.spec.getWithValues(d)

    def __rich_repr__(self):
        yield "values", self.values

    def __str__(self):
        return f"values: {self.values}"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.values)

    def getAllByName(self, name):
        return {(x, y): v for x, u in self for y, v in u if y == name}


# def mergeAnalyzerValues(first, *rest):
#     while rest:
#
# def moduleValues(first, *rest):
#     while rest:


@define
class ParameterSpec:
    default_value: Any | None = None
    possible_values: Collection | None = None
    tags: set[str] = field(factory=set)
    param_type: type | None = None


@define
class NodeParameterSpec:
    node_id: str
    parameter_spec: ParameterSpec


@define
class ModuleParameterSpec:
    param_specs: dict[str, ParameterSpec] = field(factory=dict)

    def getTags(self, *tags):
        return {
            x: y for x, y in self.param_specs.items() if any(t in y.tags for t in tags)
        }

    def __getitem__(self, key):
        return self.param_specs[key]

    def getWithValues(self, values: dict[str, Any]):
        ret = {}
        for name, spec in self.param_specs.items():
            if name in values:
                v = values[name]
                if (spec.possible_values is None or v in spec.possible_values) and (
                    spec.param_type is None or isinstance(v, spec.param_type)
                ):
                    ret[name] = values[name]
                else:
                    raise RuntimeError(
                        f"Value {v} not in the list of possible values for parameter {name}. Allowed values are {spec.possible_values}"
                    )
            else:
                if spec.default_value is None:
                    raise RuntimeError(
                        f"Must provide a value for {spec} -- {name} with no default value"
                    )
                ret[name] = spec.default_value
        return ModuleParameterValues(frozenset(ret.items()), self)


@define
class PipelineParameterSpec:
    node_specs: dict[str, ModuleParameterSpec]

    def __getitem__(self, key):
        return self.node_specs[key]

    def __setitem__(self, key, value):
        if key in self.node_specs:
            raise RuntimeError()
        self.node_specs[key] = value

    def getWithValues(self, values: dict[str, dict[str, Any]]):
        ret = {}
        for nid, spec in self.node_specs.items():
            if nid in values:
                ret[nid] = spec.getWithValues(values[nid])
            else:
                ret[nid] = spec.getWithValues({})
        return PipelineParameterValues(frozenset(ret.items()), self)

    def getTags(self, *tag):
        tags = {x: y.getTags(*tag) for x, y in self.node_specs.items()}
        return tags


def toTuples(d):
    return {(x, y): v for x, s in d.items() for y, v in s.items()}


def fromTuples(d):
    ret = defaultdict(dict)
    for (k1, k2), v in d.items():
        ret[k1][k2] = v
    return dict(ret)


def buildCombos(spec, tag):
    ret = []
    tup = toTuples(spec.getTags(tag))
    central = {k: v.default_value for k, v in tup.items()}
    for k, v in tup.items():
        for p in v.possible_values:
            if p == "central":
                continue
            c = copy.deepcopy(central)
            c[k] = p
            ret.append(c)

    ret = [fromTuples(x) for x in ret]

    return ret


def buildVariations(spec, metadata=None):
    weights = buildCombos(spec, "weight_variation")
    shapes = buildCombos(spec, "shape_variation")
    all_vars = [{}] + weights + shapes
    return all_vars


@define(frozen=True)
class ModuleProvenance:
    name: str
    parameters: ModuleParameterValues
    column_key: int


@define
class MetadataExpr(abc.ABC):
    @abc.abstractmethod
    def __call__(self, metadata): ...


@define
class MetadataFieldMatch(MetadataExpr):
    pattern: str

    def __call__(self, metadata):
        return patternMatch(pattern, metadata)


@define
class FileResource:
    path: str


@define
class AnalyzerModule(abc.ABC):
    __cache: dict = field(factory=dict, init=False, repr=False)

    # should_run: MetadataExpr | bool = True

    @abc.abstractmethod
    def run(
        self, columns, params: ModuleParameterValues
    ) -> tuple[ColumnView, list[AnalyzerResult | ModuleAddition]]:
        pass

    @abc.abstractmethod
    def inputs(self, metadata): ...

    @abc.abstractmethod
    def outputs(self, metadata): ...

    @abc.abstractmethod
    def neededResources(self, metadata): ...

    def getParameterSpec(self) -> ModuleParameterSpec:
        return ModuleParameterSpec()

    def preloadForMeta(self, metadata):
        pass

    def getColumnProvenance(self, columns):
        if isinstance(columns, ColumnView):
            return columns.getKeyForColumns(self.inputs(columns.metadata))
        else:
            return frozenset(self.getColumnProvenance(x) for x in (columns or []))

    def neededResources(self, metadata):
        return []

    def getKey(self, columns, params):
        ret = hash((self.name(), params, self.getColumnProvenance(columns)))
        return ret

    def __runNoInputs(self, params):
        key = self.getKey(None, params)
        logger.info(f"Execution key is {key}")
        logger.info(f"Cached keys are {list(self.__cache)}")
        if key in self.__cache:
            logger.info(f"Found key, using cached result")
            cached_cols, r = self.__cache[key]
            return cached_cols, r
        logger.info(f"Did not find cached result, running module {self}")
        ret = self.run(None, params)
        self.__cache[key] = ret
        return ret[0].copy(), ret[1]

    def __runStandard(self, columns, params):
        orig_columns = columns
        columns = columns.copy()
        key = self.getKey(columns, params)
        logger.info(f"Execution key is {key}")
        logger.info(f"Cached keys are {list(self.__cache)}")
        if key in self.__cache:
            logger.info(f"Found key, using cached result")
            cached_cols, r, internal = self.__cache[key]
            outputs = self.outputs(columns.metadata)
            # if "INTERNAL_USE" in cached_cols.fields:
            #     outputs.append(Column(("INTERNAL_USE",)))
            if outputs == EVENTS:
                return cached_cols, r
            outputs += internal
            columns.addColumnsFrom(cached_cols, outputs)
            columns.pipeline_data = cached_cols.pipeline_data
            return columns, r
        logger.info(f"Did not find cached result, running module {self}")
        with (
            columns.useKey(key),
            columns.allowedInputs(self.inputs(columns.metadata)),
            columns.allowedOutputs(self.outputs(columns.metadata)),
        ):
            cols, res = self.run(columns, params)
            internal = cols.updatedColumns(orig_columns, Column("INTERNAL_USE"))
        self.__cache[key] = (cols, res, internal)
        return cols, res

    def __runMulti(self, columns, params):
        orig_columns = columns
        columns = [(x, y.copy()) for x, y in columns]
        just_cols = [x[1] for x in columns]
        key = self.getKey(just_cols, params)
        logger.info(f"Execution key is {key}")
        logger.info(f"Cached keys are {list(self.__cache)}")
        if key in self.__cache:
            logger.info(f"Found key, using cached result")
            ret = self.__cache[key]
            # columns.addColumnsFrom(cached_cols, self.ouputs(columns.metadata))
            return ret
        logger.info(f"Did not find cached result, running module {self}")
        with contextlib.ExitStack() as stack:
            for c in just_cols:
                stack.enter_context(c.useKey(key))
                stack.enter_context(c.allowedOutputs(self.outputs(c.metadata)))
                stack.enter_context(c.allowedInputs(self.inputs(c.metadata)))
            ret = self.run(columns, params)

        self.__cache[key] = ret
        return ret

    def __call__(self, columns, params):
        logger.info(f"Running analyzer module {self}")
        # if not self.should_run(columns.metadata):
        #     returncolumns, {}
        if isinstance(columns, ColumnView):
            return self.__runStandard(columns, params)
        elif isinstance(columns, list):
            return self.__runMulti(columns, params)
        elif columns is None:
            return self.__runNoInputs(params)
        else:
            raise RuntimeError()

    @classmethod
    def name(cls):
        return cls.__name__


class Node:
    def __init__(
        self,
        node_id: str,
        analyzer_module: AnalyzerModule,
        parent: Node | None = None,
    ):
        self.node_id = node_id
        self.parent = parent
        self.analyzer_module = analyzer_module
        self.request_parameter_runs = []

    def getModuleSpec(self):
        return self.analyzer_module.getParameterSpec()

    def __eq__(self, other):
        return (
            self.node_id == other.node_id
            and self.analyzer_module == other.analyzer_module
        )

    def __call__(self, columns, params):
        params = params[self.node_id]
        return self.analyzer_module(columns, params)

    def __rich_repr__(self):
        yield "node_id", self.node_id
        yield "module_id", id(self.analyzer_module)
        yield "analyzer_module", self.analyzer_module


@define
class ModuleAddition:
    analyzer_module: AnalyzerModule
    run_builder: Any | None = None
    this_module_parameters: dict = field(factory=dict)
    # parameter_runs: list[PipelineParameterValues]


# @define
# class ProvenanceTrie:
#     elements: dict[str, ColumnTrie | int] = field(factory=dict)


def setColumn(events, column, value):
    column = Column(column)
    if len(column) == 1:
        return ak.with_field(events, value, column.fields)
    head, *rest = tuple(column)
    if head not in events.fields:
        for c in reversed(list(rest)):
            value = ak.zip({c: value})
        return ak.with_field(events, value, head)
    else:
        return ak.with_field(events, setColumn(events[head], Column(rest), value), head)


@define
class ColumnView:
    INTERNAL_USE_COL: ClassVar[Column] = Column("INTERNAL_USE")
    _events: Any
    _column_provenance: dict[Column, int]
    backend: EventBackend
    _current_provenance: int | None = None
    _allowed_inputs: ColumnCollection | None = None
    _allowed_outputs: ColumnCollection | None = None
    _allow_filter: bool = True
    metadata: Any | None = None
    pipeline_data: dict[str, Any] = field(factory=dict)

    @property
    def fields(self):
        return self._events.fields

    def updatedColumns(self, old, limit):
        cols_to_consider = {
            x: y for x, y in self._column_provenance.items() if limit.contains(x)
        }
        old_to_consider = {
            x: y for x, y in old._column_provenance.items() if limit.contains(x)
        }
        return [
            x
            for x, y in cols_to_consider.items()
            if x not in old_to_consider or y != old_to_consider[x]
        ]

    def copy(self):
        return ColumnView(
            events=copy.copy(self._events),
            column_provenance=copy.copy(self._column_provenance),
            metadata=copy.copy(self.metadata),
            pipeline_data=copy.deepcopy(self.pipeline_data),
            backend=self.backend,
        )

    @staticmethod
    def fromEvents(events, metadata, backend, provenance):
        return ColumnView(
            events=events,
            column_provenance={x: provenance for x in getAllColumns(events)},
            current_provenance=provenance,
            backend=backend,
            metadata=metadata,
        )

    def getKeyForColumns(self, columns):
        """
        Get an excecution key for the column.
        Returns a hash dependent on the provenance of all the columns contains in the input.
        """
        ret = []
        for column in columns:
            for c, v in self._column_provenance.items():
                if column.contains(c):
                    ret.append((c, v))

        logger.info(f"Relevant columns for {columns} are :\n {ret}")
        return hash((freeze(self.metadata), freeze(self.pipeline_data), freeze(ret)))

    def __setitem__(self, column, value):
        column = Column(column)
        if (
            self._allowed_outputs is not None
            and not ColumnView.INTERNAL_USE_COL.contains(column)
            and not self._allowed_outputs.contains(column)
        ):
            raise RuntimeError(
                f"Column {column} is not in the list of outputs {self._allowed_outputs}"
            )
        self._events = setColumn(self._events, column, value)
        self._column_provenance[column] = self._current_provenance
        all_columns = getAllColumns(value, column)
        for c in all_columns:
            self._column_provenance[c] = self._current_provenance
            logger.info(
                f"Adding column {c} to events with provenance {self._current_provenance}"
            )
        for c in self._column_provenance:
            if c.contains(column):
                self._column_provenance[c] = self._current_provenance
                logger.info(
                    f"Updating parent column {c} to events with provenance {self._current_provenance}"
                )

    def __getitem__(self, column):
        column = Column(column)
        if self._allowed_inputs is not None and not self._allowed_inputs.contains(
            column
        ):
            raise RuntimeError(
                f"Column {column} is not in the list of inputs {self._allowed_inputs}"
            )
        return column.extract(self._events)

    def addColumnsFrom(self, other, columns):
        for column in columns:
            with self.useKey(other._column_provenance[column]):
                self[column] = other[column]
            # self._setIndividualColumnWithProvenance(
            #     column, other[column], other._column_provenance[column]
            # )

    def filter(self, mask):
        if not self._allow_filter:
            raise RuntimeError()
        new_cols = copy.copy(self)
        new_cols._events = new_cols._events[mask]
        for c in self._column_provenance:
            self._column_provenance[c] = self._current_provenance
        return new_cols

    @contextlib.contextmanager
    def useKey(self, provenance: ModuleProvenance):
        old_provenance = self._current_provenance
        self._current_provenance = provenance
        yield
        self._current_provenance = old_provenance

    @contextlib.contextmanager
    def allowedInputs(self, columns):
        columns = ColumnCollection(columns)
        old_inputs = self._allowed_inputs
        self._allowed_inputs = columns
        yield
        self._allowed_inputs = old_inputs

    @contextlib.contextmanager
    def allowedOutputs(self, columns):
        columns = ColumnCollection(columns)
        old_outputs = self._allowed_outputs
        self._allowed_outputs = columns
        yield
        self._allowed_outputs = old_outputs

    @contextlib.contextmanager
    def allowFilter(self, allow):
        old_allow = self._allowed_filter
        self._allow_filter = allow
        yield
        self._allow_filter = old_allow


def mergeColumns(column_views):
    ret = copy.copy(column_views[0])
    for other in column_views[1:]:
        ret = ret.addColumnsFrom(other)
    return ret


@define
class GoodJetMaker(AnalyzerModule):
    input_col: Column
    output_col: Column
    min_pt: float = 30.0
    max_abs_eta: float = 2.4
    include_pu_id: bool = False
    include_jet_id: bool = False

    def run(self, columns, params):
        jets = columns[self.input_col]
        good_jets = jets[(jets.pt > self.min_pt) & (abs(jets.eta) < self.max_abs_eta)]
        columns[self.output_col] = good_jets
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class BQuarkMaker(AnalyzerModule):
    input_jets: Column
    output_col: Column
    working_point: float = 0.5

    __corrections: dict[str, Any] = field(factory=dict, eq=False)

    def run(self, columns, params):
        disc = columns[self.input_jets].btagDeepFlavB
        bjets = columns[self.input_jets][disc > self.working_point]
        columns[self.output_col] = bjets
        return columns, []

    def preloadForMeta(self, metdata):
        pass

    def inputs(self, metadata):
        return [self.input_jets]

    def outputs(self, metadata):
        return [self.output_col]


@define
class SomeScaleFactor(AnalyzerModule):
    input_jets: Column

    def run(self, columns, params):
        variation = params["variation"]
        j = columns[self.input_jets]
        if variation == "central":
            columns["Weight.some_scale_factor"] = ak.ones_like(j.pt)
        elif variation == "down":
            columns["Weight.some_scale_factor"] = 0.5 * ak.ones_like(j.pt)
        elif variation == "up":
            columns["Weight.some_scale_factor"] = 2 * ak.ones_like(j.pt)

        return columns, []

    def getParameterSpec(self):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=["central", "up", "down"],
                    tags={
                        "weight_variation",
                    },
                ),
                "other_parameter": ParameterSpec(
                    default_value=1,
                    possible_values=[1, 2, 3],
                    tags={},
                ),
            }
        )

    def inputs(self, metadata):
        return [self.input_jets]

    def outputs(self, metadata):
        return [Column("Weight.some_scale_factor")]


@define
class SomeOtherScaleFactor(AnalyzerModule):
    input_jets: Column

    def run(self, columns, params):
        variation = params["variation"]
        j = columns[self.input_jets]
        if variation == "central":
            columns["Weight.some_scale_factor"] = ak.ones_like(j.pt)
        elif variation == "down":
            columns["Weight.some_scale_factor"] = 0.5 * ak.ones_like(j.pt)
        elif variation == "up":
            columns["Weight.some_scale_factor"] = 2 * ak.ones_like(j.pt)

        return columns, []

    def getParameterSpec(self):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=["central", "up", "down"],
                    tags={
                        "weight_variation",
                    },
                )
            }
        )

    def inputs(self, metadata):
        return [self.input_jets]

    def outputs(self, metadata):
        return [Column("Weight.some_scale_factor")]


@define
class SomeShapeScaleFactor(AnalyzerModule):
    input_jets: Column
    output_jets: Column

    def run(self, columns, params):
        variation = params["variation"]
        columns[self.output_jets] = columns[self.input_jets]
        return columns, []

    def getParameterSpec(self):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=[
                        "central",
                        "shape_up_1",
                        "shape_down_1",
                        # "shape_up_2",
                        # "shape_down_2",
                        # "shape_down_3",
                        # "shape_down_4",
                        # "shape_down_5",
                        # "shape_down_6",
                        # "shape_down_7",
                        # "shape_down_8",
                        # "shape_down_9",
                        # "shape_down_10",
                        # "shape_down_11",
                        # "shape_down_12",
                        # "shape_down_13",
                        # # "shape_down_14",
                        # "shape_down_15",
                        # "shape_down_16",
                        # "shape_down_17",
                        # "shape_down_18",
                        # "shape_down_20",
                    ],
                    tags={
                        "shape_variation",
                    },
                )
            }
        )

    def inputs(self, metadata):
        return [self.input_jets]

    def outputs(self, metadata):
        return [self.output_jets]


@define
class NObjFilter(AnalyzerModule):
    selection_name: str
    input_col: Column
    min_count: int | None = None
    max_count: int | None = None

    def run(self, columns, analyzer, **kwargs):
        objs = columns[self.input_col]
        count = ak.num(objs, axis=1)
        sel = None
        if self.min_count is not None:
            sel = count >= self.min_count
        if self.max_count is not None:
            if sel is not None:
                sel = sel & (count <= self.max_count)
            else:
                sel = count <= self.max_count
        columns["Selection", self.selection_name] = sel
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]


@define
class LoadColumns(AnalyzerModule):
    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return None

    def getParameterSpec(self):
        return ModuleParameterSpec(
            {
                "chunk": ParameterSpec(param_type=EventSource, tags={"column-input"}),
            }
        )

    def run(self, columns, params):
        key = self.getKey(None, params)
        ret = params["chunk"].loadEvents(
            "coffea-virtual", view_kwargs=dict(metadata=None, provenance=key)
        )
        return ret, []


def getPipelineSpecs(pipeline):
    ret = {}
    for node in pipeline:
        ret[node.node_id] = node.getModuleSpec()
    return PipelineParameterSpec(ret)


@define
class HLTPass(AnalyzerModule):
    out_name: str
    hlt_trigger_names: list[str]

    def run(self, columns, params):
        ret = columns["HLT", self.hlt_trigger_names[0]]
        for name in self.hlt_trigger_names[1:]:
            ret &= columns["HLT"][name]
        columns["Selection", self.out_name] = ret
        return columns, []

    def init(self, config):
        pass

    def inputs(self, metadata):
        return [Column(("HLT", x)) for x in self.hlt_trigger_names]

    def outputs(self, metadata):
        return [Column(("Selection", self.out_name))]


# @register_module(
#     ["HLT"], dict(out_name=field(type=str), hlt_trigger_names=field(type=list[str]))
# )
# def HLTPass(self, columns, params):
#     ret = columns["HLT"][self.hlt_trigger_names[0]]
#     for name in self.hlt_trigger_names[1:]:
#         ret &= columns["HLT"][name]
#     columns["Selection", self.out_name] = ret
#     return columns, {}


@define
class SelectOnColumns(AnalyzerModule):
    column_names: list[str]

    def run(self, columns, params):
        cols = self.inputs(columns.metadata)
        ret = columns[cols[0]]
        for name in cols[1:]:
            ret = ret & columns[name]
        columns = columns.filter(ret)
        return columns, []

    def init(self, config):
        pass

    def inputs(self, metadata):
        return [Column("Selection") + x for x in self.column_names]

    def outputs(self, metadata):
        return EVENTS


@define(frozen=True)
class IntegerAxis:
    name: str
    start: int
    stop: int
    unit: str | None = None

    def toHist(self):
        return hist.axis.Integer(self.start, self.stop, name=self.name)


@define(frozen=True)
class RegularAxis:
    name: str
    bins: int
    start: float
    stop: float
    unit: str | None = None

    def toHist(self):
        return hist.axis.Regular(self.bins, self.start, self.stop, name=self.name)


Axis = RegularAxis | IntegerAxis


@define(frozen=True)
class CategoryDesc:
    column: Column
    axis: Any


@define
class HistogramBuilder(AnalyzerModule):
    product_name: str
    columns: list[Column]
    axes: list[Axis]
    central_name: str = "central"
    storage: str = "weight"
    mask_col: Column | None = None

    @staticmethod
    def transformToFill(fill_data, per_event_value, mask=None):
        """
        Perform transformations to bring fill data to correct shape
        """
        if mask is None:
            if fill_data.ndim == 2:
                fill_data = ak.ones_like(fill_data, dtype=np.int32)
                fill_data = ak.fill_none(fill_data, 0)
                r = ak.flatten(fill_data * per_event_value)
                return r
            else:
                return per_event_value

        if mask.ndim == 1 and not (fill_data is None) and fill_data.ndim == 2:
            fill_data = ak.ones_like(fill_data, dtype=np.int32)
            fill_data = ak.fill_none(fill_data, 0)
            return ak.flatten(fill_data * per_event_value[mask])

        elif mask.ndim == 2:
            return ak.flatten((ak.ones_like(mask) * per_event_value)[mask])

        else:
            return per_event_value[mask]

    @staticmethod
    def maybeFlatten(data):
        if data.ndim == 2:
            return ak.flatten(data)
        else:
            return data

    @staticmethod
    def fillHistogram(
        histogram,
        cat_values,
        fill_data,
        weight=None,
        variation="central",
        mask=None,
    ):
        all_values = (
            [variation]
            + cat_values
            + [HistogramBuilder.maybeFlatten(x) for x in fill_data]
        )
        if weight is not None:
            histogram.fill(*all_values, weight=weight)
        else:
            histogram.fill(*all_values)
        return histogram

    @staticmethod
    def create(backend, categories, axes, storage):
        variations_axis = hist.axis.StrCategory([], name="variation", growth=True)
        all_axes = (
            [variations_axis]
            + [x.axis.toHist() for x in categories]
            + [x.axes.toHist() for x in axes]
        )
        if backend == EventBackend.coffea_dask:
            histogram = dah.Hist(*all_axes, storage=storage)
        else:
            histogram = hist.Hist(*all_axes, storage=storage)
        return histogram

    def run(self, column_sets, params):
        backend = column_sets[0][1].backend
        pipeline_data = column_sets[0][1].pipeline_data
        categories = pipeline_data.get("categories", {})
        histogram = HistogramBuilder.create(
            backend, categories, self.axes, self.storage
        )
        for cset in column_sets:
            params = cset[0]
            noncentral = {
                k: v
                for k, v in params.getAllByName("variation").items()
                if v != self.central_name
            }
            if len(noncentral) == 0:
                variation_name = "central"
            elif len(noncentral) == 1:
                variation_name = next(iter(noncentral.values()))
            else:
                raise RuntimeError(f"Multiple active systematics {noncentral}")
            columns = cset[1]
            data_to_fill = [columns[x] for x in self.columns]
            represenative = data_to_fill[0]
            mask = None
            if self.mask_col is not None:
                mask = columns[self.mask_col]

            if "Weights" in columns.fields:
                weights = columns["Weights"]
                total_weight = ak.prod([weights[x] for x in weights.fields], axis=1)
                total_weight = HistogramBuilder.transformToFill(
                    represenative, weight, mask
                )
            else:
                total_weight = None

            cat_to_fill = [
                HistogramBuilder.transformToFill(represenative, columns[x.column], mask)
                for x in categories
            ]
            HistogramBuilder.fillHistogram(
                histogram,
                cat_to_fill,
                data_to_fill,
                weight=total_weight,
                variation=variation_name,
            )

        return None, [Histogram(name=self.product_name, histogram=histogram)]

    def inputs(self, metadata):
        return [*self.columns, Column("Categories"), Column("Weights")]

    def outputs(self, metadata):
        return []


def makeHistogram(product_name, columns, axes, data, description, want_variations=None):
    if not isinstance(data, (list, tuple)):
        data = [data]
        axes = [axes]

    names = []
    for i, d in enumerate(data):
        name = Column(f"INTERNAL_USE.auto-col-{product_name}-{i}")
        names.append(name)
        columns[name] = d

    b = HistogramBuilder(product_name, names, axes, axes)
    return ModuleAddition(b, buildVariations)


def addCategory(columns, name, data, axis):
    col = Column(fields=("Categories", name))
    columns[col] = data
    to_add = CategoryDesc(column=col, axis=axis)
    if "categories" not in columns.pipeline_data:
        columns.pipeline_data["categories"] = []
    columns.pipeline_data["categories"].append(to_add)
    return columns


@define
class NObjCategory(AnalyzerModule):
    input_col: Column
    category_name: str
    ax_min: int = 0
    ax_max: int = 10

    def run(self, columns, params):
        addCategory(
            columns,
            self.category_name,
            ak.num(columns[self.input_col]),
            IntegerAxis(self.category_name, self.ax_min, self.ax_max),
        )
        return columns, []

    def init(self, config):
        pass

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Categories", self.category_name))]


@define
class TestHistogram(AnalyzerModule):
    col: Column

    def run(self, columns, params):
        j = columns[self.col]
        h = makeHistogram(
            "pt_1", columns, hist.axis.Regular(10, 0, 10), j.pt, None, None
        )
        return columns, [h]

    def init(self, config):
        pass

    def inputs(self, metadata):
        return [self.col]

    def outputs(self, metadata):
        return []


class Analyzer:
    def __init__(self):
        self.all_modules = []
        self.base_pipelines: dict[str, list[Node]] = {}

    def initModules(self, metadata):
        for m in self.all_modules:
            m.preloadForMeta(metadata)

    def getUniqueNode(self, pipeline, module):
        base = module.name()
        to_use = base
        i = 0
        while any(
            x.node_id == to_use
            for x in it.chain(pipeline, *self.base_pipelines.values())
        ):
            i += 1
            to_use = base + "-" + str(i)
        found = next((x for x in self.all_modules if x == module), None)
        if found is not None:
            return Node(to_use, found)
        else:
            n = Node(to_use, module)
            self.all_modules.append(module)
            return n

    def addPipeline(self, name, pipeline):
        ret = []
        for module in pipeline:
            ret.append(self.getUniqueNode(ret, module))
        self.base_pipelines[name] = ret

    def runPipelineWithParameters(
        self,
        columns,
        pipeline,
        params,
        pipeline_name,
        freeze_pipeline=False,
        handle_results=True,
    ):
        orig_columns = columns.copy()
        complete_pipeline = []
        to_add = deque(pipeline)
        if handle_results:
            result_container = ResultContainer(pipeline_name)
        else:
            result_container = None

        while to_add:
            head = to_add.popleft()
            complete_pipeline.append(head)
            current_spec = getPipelineSpecs(complete_pipeline)
            logger.info(f"Running node {head}")
            columns = columns.copy()
            columns, results = head(columns, params)
            logger.info(f"Node produced {len(results)} results")

            if not handle_results:
                continue
            results = deque(results)

            while results:
                res = results.popleft()
                if isinstance(res, AnalyzerResult):
                    result_container.addResult(res)
                elif isinstance(res, ModuleAddition) and not freeze_pipeline:
                    module = res.analyzer_module
                    if not res.run_builder:
                        logger.info(f"Adding new module {module} to pipeline")
                        to_add.appendleft(self.getUniqueNode(complete_pipeline, module))
                    else:
                        everything = []
                        logger.info(f"Running multi-parameter pipeline")
                        param_dicts = res.run_builder(current_spec, columns.metadata)
                        to_run = [params.withNewValues(x) for x in param_dicts]

                        for params_set in to_run:
                            c, _ = self.runPipelineWithParameters(
                                orig_columns,
                                complete_pipeline,
                                params_set,
                                pipeline_name="NONE",
                                freeze_pipeline=True,
                                handle_results=False,
                            )
                            everything.append((params_set, c))
                        logger.info(
                            f"Running node {module} with {len(everything)} parameter sets"
                        )
                        _, r = module(
                            everything, params.withNewValues(res.this_module_parameters)
                        )
                        results.extendleft(r)
                else:
                    raise RuntimeError(
                        f"Invalid object type returned from analyzer module."
                    )
        return columns, result_container

    def __rich_repr__(self):
        yield "all_modules", self.all_modules
        yield "base_pipelines", self.base_pipelines

    def run(self, chunk, pipelines=None):
        pipelines = pipelines or list(self.base_pipelines)
        loader = LoadColumns()
        columns, _ = loader(
            None,
            ModuleParameterValues(
                frozenset((("chunk", chunk),)),
            ),
        )
        result_container = ResultContainer("ChunkResult")
        for k, pipeline in self.base_pipelines.items():
            if k not in pipelines:
                continue
            columns = columns.copy()
            spec = getPipelineSpecs(pipeline)
            vals = spec.getWithValues({})
            _, ret = self.runPipelineWithParameters(
                columns, pipeline, vals, pipeline_name=k
            )
            result_container.addResult(ret)
        return result_container


# module_unstr = converter.get_unstructure_hook(AnalyzerModule)
# module_str = converter.get_structure_hook(AnalyzerModule)
#
# @converter.register_structure_hook
# def deserialize(data, t) -> Analyzer:
#     analyzer = t()
#     for k, v in data.items():
#         analyzer.addPipeline(k, converter.structure(v, list[AnalyzerModule]))
#     return analyzer
#
#
# @converter.register_unstructure_hook
# def serialize(analyzer: Analyzer):
#     return {
#         x: converter.unstructure([z.analyzer_module for z in y], list[AnalyzerModule])
#         for x, y in analyzer.base_pipelines.items()
#     }


def naiveIntersect(s1, s2):
    for i in s1:
        for j in s2:
            if i.intersects(j):
                return True
    return False


def main2():
    rf = [RootFile(f"{i}" * 40, "nano_dy.root") for i in range(10000)]
    chunked = []
    for i in rf:
        for j in range(0, 5000000, 10000):
            chunked.append(ChunkedRootFile(i, j, j + 100000))

    s1 = random.sample(chunked, 6000)
    s2 = random.sample(chunked, 100)

    start_time = timeit.default_timer()
    ret = naiveIntersect(s1, s2)
    elapsed = timeit.default_timer() - start_time


def main():
    f = RootFile("1", "signal_312_900_600_plus.root")
    # f = RootFile("1", "nano_dy.root")
    chunks = f.chunks(100000000000)
    load = LoadColumns()
    # hlt_pass = HLTPass("pass_trigger", ["PFHT1050"])
    # good_jets = GoodJetMaker(Column("Jet"), Column("GoodJet"))
    b_jets = BQuarkMaker(Column("CorrectedJets"), Column("MedBJet"))
    # b_jets2 = BQuarkMaker(Column("CorrectedJets"), Column("TightBJet"))
    n_obj_filter = NObjFilter("med_b_jets", Column("MedBJet"), None, 1)
    # trigger_filter = SelectOnColumns(["pass_trigger", "med_b_jets"])
    # smaller_filter = SelectOnColumns(["pass_trigger"])
    # some_sf = SomeScaleFactor(Column("GoodJet"))
    test_h = TestHistogram(Column("CorrectedJets"))

    pipeline = [
        SomeShapeScaleFactor(Column("Jet"), Column("CorrectedJets")),
        # HLTPass("pass_trigger", ["PFHT1050"]),
        b_jets,
        n_obj_filter,
        SelectOnColumns(["med_b_jets"]),
        NObjCategory(Column("CorrectedJets"), "Njets"),
        test_h,
    ]

    # pipeline2 = [
    #     SomeShapeScaleFactor(Column("Jet"), Column("CorrectedJets")),
    #     # HLTPass("pass_trigger", ["PFHT1050"]),
    #     b_jets,
    #     n_obj_filter,
    #     test_h,
    # ]

    analyzer = Analyzer()
    analyzer.addPipeline("pipe1", pipeline)
    analyzer.addPipeline("pipe2", pipeline)
    analyzer.initModules(None)
    p1 = analyzer.base_pipelines["pipe1"]
    spec = getPipelineSpecs(p1)
    vals = spec.getWithValues({"LoadColumns": {"chunk": chunks[0]}})
    spec.getTags("column-input")
    ret = analyzer.run(chunks[0])
    print(ret)


if __name__ == "__main__":
    main()
