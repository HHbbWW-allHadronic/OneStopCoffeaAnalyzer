from __future__ import annotations
import numbers
import itertools as it
import dask_awkward as dak
import hist

from collections.abc import Collection
from collections import deque

from pydantic import BaseModel, model_validator, RootModel, Field, ConfigDict
from dataclasses import dataclass, field

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
    level="NOTSET",  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    format=FORMAT,
    handlers=[RichHandler()],
)

# Get a logger instance
log = logging.getLogger("my_app")


logger = logging.getLogger(__name__)


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


@dataclass
class Column:
    fields: tuple[str, ...]

    def __init__(self, data):
        if isinstance(data, str):
            self.fields = Column.fieldsFromString(data)
        elif isinstance(data, Column):
            self.fields = data.fields
        else:
            self.fields = data

    @staticmethod
    def fieldsFromString(data):
        return tuple(data.split("."))

    def parts(self):
        return tuple(self.path)

    def contains(self, other):
        if len(self) > len(other):
            return False
        return other[: len(self)] == self

    def commonParent(self, other):
        l = []
        for x, y in zip(self.iterParts(), other.iterParts()):
            if x == y:
                l.append(x)
            else:
                break
        ret = Column(tuple(l))
        return ret

    def extract(self, events):
        return ft.reduce(lambda x, y: x[y], self.fields, events)

    def iterParts(self):
        return iter(self.fields)

    def __eq__(self, other):
        return self.fields == other.fields

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, key):
        return Column(self.fields.__getitem__(key))

    def __add__(self, other):
        return Column(self.fields + Column(other).fields)

    def __radd__(self, other):
        return Column(Column(other).fields + self.fields)

    def __iter__(self):
        return (Column(x) for x in self.fields)

    def __hash__(self):
        return hash(self.fields)


class AnalyzerResult(abc.ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    identity: str

    @abc.abstractmethod
    def __iadd__(self, other):
        pass

    @abc.abstractmethod
    def iscale(self, value):
        pass

    def finalize(self, finalizer):
        return

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def scale(self, value):
        ret = copy.deepcopy(self)
        return ret.iscale(value)


class ResultContainer(AnalyzerResult):
    results: dict[str, AnalyzerResult] = Field(default_factory=dict)

    def __iadd__(self, other):
        for k in self.results:
            self.results[k] += other.results[k]

    def addResult(self, result):
        name = result.name
        if name in self.results:
            raise KeyError(
                f"Result container already contains a result with name {name}"
            )
        self.results[name] = result

    def getResult(self, name):
        return self.results[name]

    def iscale(self, value):
        for k in self.results:
            self.results[k].iscale(value)

    def finalize(self, finalizer):
        self.results = finalizer(results)
        for result in self.results.values():
            result.finalize(finalizer)


class Histogram(AnalyzerResult):
    histogram: hist.Hist

    def __iadd__(self, other):
        self.hist += other.hist

    def iscale(self, value):
        self.hist *= value


class ScalableArray(AnalyzerResult):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate(self.array, other.array, axis=0)

    def iscale(self, value):
        self.array *= value

    def finalize(self, finalizer):
        if isinstance(self.array, np.ndarray):
            pass
        elif isinstance(self.array, ak.Array):
            self.array = self.array.to_numpy()
        elif isinstance(self.array, dak.Array):
            self.array = dask.compute(self.array).to_numpy()


class RawArray(AnalyzerResult):
    array: ak.Array | dak.Array | np.ndarray

    def __iadd__(self, other):
        if isinstance(self.array, np.ndarray):
            self.array = np.concatenate(self.array, other.array, axis=0)

    def iscale(self, value):
        pass

    def finalize(self, finalizer):
        if isinstance(self.array, np.ndarray):
            pass
        elif isinstance(self.array, ak.Array):
            self.array = self.array.to_numpy()
        elif isinstance(self.array, dak.Array):
            self.array = dask.compute(self.array).to_numpy()


Scalar = dak.Scalar | numbers.Real


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


@dataclass
class ColumnCollection:
    columns: set[Column]

    def __iter__(self):
        return iter(columns)

    def contains(self, other: Column):
        # breakpoint()
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
        return ret
    else:
        return {cur_col}


NanoAODSchema.warn_missing_crossrefs = False


@dataclass(frozen=True)
class ModuleParameterValues:
    param_values: frozenset[tuple[str, Any]]
    spec: ModuleParameterSpec | None = field(
        compare=False, default_factory=lambda: None
    )

    @ft.lru_cache()
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


@dataclass(frozen=True)
class AnalyzerParameterValues:
    values: frozenset[tuple[str, ModuleParameterValues]]
    spec: AnalyzerParameterSpec


    def __hash__(self):
        return hash(self.values)

    def __getitem__(self, key):
        found = next(x[1] for x in self.values if x[0] == key)
        return ModuleParameterValues(found.param_values, self)

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
        
        
        


@dataclass
class ParameterSpec:
    default_value: Any | None = None
    possible_values: Collection | None = None
    tags: set[str] = field(default_factory=set)


@dataclass
class NodeParameterSpec:
    node_id: str
    parameter_spec: ParameterSpec


@dataclass
class ModuleParameterSpec:
    param_specs: dict[str, ParameterSpec] = field(default_factory=dict)

    def getTags(self, tag):
        return {x:y for x,y in self.param_specs.items() if tag in y.tags}

    def __getitem__(self, key):
        return self.param_specs[key]

    def getWithValues(self, values: dict[str, Any]):
        ret = {}
        for name, spec in self.param_specs.items():
            if name in values:
                v = values[name]
                if spec.possible_values is None or v in spec.possible_values:
                    ret[name] = values[name]
                else:
                    raise RuntimeError()
            else:
                if spec.default_value is None:
                    raise RuntimeError(
                        f"Must provide a value for {spec} -- {name} with no default value"
                    )
                ret[name] = spec.default_value
        return ModuleParameterValues(frozenset(ret.items()), self)


@dataclass
class AnalyzerParameterSpec:
    node_specs: dict[str, ModuleParameterSpec]

    def getWithValues(self, values: dict[str, dict[str, Any]]):
        ret = {}
        for nid, spec in self.node_specs.items():
            if nid in values:
                ret[nid] = spec.getWithValues(values[nid])
            else:
                ret[nid] = spec.getWithValues({})
        return AnalyzerParameterValues(frozenset(ret.items()), self)

    def getTags(self, tag):
        tags = {x: y.getTags(tag) for x, y in self.node_specs.items()}
        return tags


class AnalyzerModule(abc.ABC):
    @abc.abstractmethod
    def run(self, columns, analyzer, params: ModuleParameterValues):
        pass

    # @abc.abstractmethod

    @abc.abstractmethod
    def getInputs(self) -> ColumnCollection:
        pass

    def init(self, params):
        pass

    def inputs(self) -> ColumnCollection:
        return self.getInputs()

    def getParameterSpec(self) -> ModuleParameterSpec:
        return ModuleParameterSpec()

    def baseInit(self, params):
        self.init(params)
        self._cache = {}

    def getProvenance(self, columns, params):
        ret = ModuleProvenance(
            name=self.name(),
            parameters=params,
            column_key=self.getColumnProvenance(columns),
        )
        return ret

    def getColumnProvenance(self, columns):
        if isinstance(columns, ColumnView):
            return columns.getKeyForColumns(self.inputs())
        else:
            return frozenset(self.getColumnProvenance(x) for x in (columns or []))

    def __call__(self, columns, params):
        if isinstance(columns, ColumnView):
            columns = columns.copy()
            just_cols = [columns]
        elif isinstance(columns, list):
            columns = [(x, y.copy()) for x, y in columns]
            just_cols = [x[1] for x in columns]
        else:
            just_cols = columns

        logger.info(f"Running analyzer module {self} ({id(self)})")
        prov = self.getProvenance(just_cols, params)

        logger.info(f"Provenance is {prov}")
        if prov in self._cache:
            logger.info(f"Found provenance, using cached result")
            return self._cache[prov]
        logger.info(f"Did not find provenance, running module {self.name()}")

        if columns is None:
            ret = self.run(columns, params)
        else:
            with contextlib.ExitStack() as stack:
                for c in just_cols:
                    stack.enter_context(c.useProvenance(prov))
                    stack.enter_context(c.allowedInputs(self.inputs()))
                ret = self.run(columns, params)
        self._cache[prov] = ret
        return ret

    @classmethod
    def name(cls):
        return cls.__name__


@dataclass(frozen=True)
class ModuleProvenance:
    name: str
    parameters: ModuleParameterValues
    column_key: ColumnView.Key


# class Node:
#     def __init__(
#         self,
#         analyzer_module: AnalyzerModule,
#         previous: Node | None = None,
#     ):
#         self.node_id = node_id
#         self.analyzer_module = analyzer_module
#         self.previous = previous
#
#     def run

# def getParameterSpec(self):
#     specs = self.analyzer_module.getParameterSpecs()
#     return NodeParameterSpec(
#         node_id=self.node_id, param_specs={x.name: x for x in specs}
#     )
#
# def getDependentParents(self):
#     inputs = self.analyzer_module.inputs()
#     previous = self.previous
#     ret = []
#     while previous is not None:
#         if previous.analyzer_module.inputs().intersect(inputs):
#             ret.append(previous)
#     return ret
#
# def getKey(self, parameters):
#     hparam = hash(parameters)
#     if hparam in self.__key_cache:
#         return self.__key_cache[hparam]
#     parents = getDependentParents()
#     my_params = parameters[self.node_id]
#     ret = hash(my_params)
#     for parent in parents:
#         ret = hash((ret, parents.getKey(parameters)))
#
#     self.__key_cache[hparam] = ret
#     return ret
#
# def executeSingle(
#     self,
#     params: ModuleParameterValues,
#     result_handler,
# ):
#
#     to_merge = []
#     for p in parents:
#         to_merge.append(self.parent.executeSingle(params))
#     my_params = parameters[self.node_id]
#     cols = mergeColumns(to_merge)
#     with cols.allowedInputs(self.analyzer_module.inputs), cols.allowedOutputs(
#         self.analyzer_module.outputs
#     ):
#         cols, results = self.analyzer_module(cols, my_params)
#     return cols
#
# def executeMulti(
#     self,
#     params: frozenset[ModuleParameterValues],
#     result_storage,
# ):
#     my_params = set(p[self.node_id] for p in params)
#     if len(my_params) != 1:
#         raise RuntimeError()
#     my_params = next(iter(my_params))
#     all_sets = []
#     for param in params:
#         to_merge = []
#         for p in parents:
#             to_merge.append(self.parent.execute(params))
#         cols = mergeColumns(to_merge)
#         all_sets.append((param, cols))
#
#     with cols.allowedInputs(self.analyzer_module.inputs), cols.allowedOutputs(
#         self.analyzer_module.outputs
#     ):
#         cols, results = self.analyzer_module(all_sets, my_params)
#     return cols
#
# def execute(
#     self,
#     params: ModuleParameterValues | list[ModuleParameterValues],
#     result_storage,
# ):
#     key = getKey(params)
#     if key in self.__key_cache:
#         return self._run_cache[key]
#     if isinstance(params, list):
#         return self.executeMulti(params, result_storage)
#     else:
#         return self.executeSingle(params, result_storage)

# def getLastDependency(self):
#     products = copy.copy(self.analyzer_module.products)
#     p = self.parent
#     ret = []
#     while products:
#         dependent_inputs = self.analyzer_module.getDependentInputs(
#             p.analyzer_module
#         )
#         if dependent_inputs:
#             ret.append(p)
#             products -= dependent_inputs
#         p = p.parent
#
#     return ret


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


@dataclass
class ModuleAddition:
    analyzer_module: AnalyzerModule
    parameter_runs: list[AnalyzerParameterValues]
    this_module_parameters: ModuleParameterValues


class Analyzer:
    def __init__(self):
        self.all_modules = []
        self.base_pipelines: dict[str, list[Node]] = {}

    def initModules(self, metadata):
        for m in self.all_modules:
            m.baseInit(metadata)

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
        start=None,
        freeze_pipeline=False,
        handle_results=True,
    ):
        complete_pipeline = []
        logger.info(f"Running pipeline {pipeline} with parameters {params}")
        to_add = deque(pipeline)
        previous = None
        if handle_results:
            result_container = ResultContainer(identity="container")

        while to_add:
            head = to_add.popleft()
            complete_pipeline.append(head)
            logger.info(f"Running node {head}")
            columns, results = head(columns, params)
            logger.info(f"Node produced {len(results)} results")
            if not handle_results:
                continue
            results = deque(results)
            while results:
                res = results.popleft()
                if isinstance(res, AnalyzerResult):
                    result_container.addResult(res)
                if isinstance(res, ModuleAddition) and not freeze_pipeline:
                    logger.info(f"Adding new node {res} to pipeline")
                    module = res.analyzer_module
                    module.baseInit(None)
                    param_req = res.parameter_runs
                    if not param_req:
                        all_results.append(results)
                        to_add.appendleft(self.getUniqueNode(complete_pipeline, module))
                    else:
                        everything = []
                        logger.info(f"Running multi-parameter pipeline")
                        for params_set in param_req:
                            c, _ = self.runPipelineWithParameters(
                                None,
                                complete_pipeline,
                                params_set,
                                freeze_pipeline=True,
                                handle_results=False,
                            )
                            everything.append((params_set, c))
                        logger.info(
                            f"Running node {res} with {len(everything)} parameter sets"
                        )
                        _, r = module(everything, res.this_module_parameters)
                        results.extendleft(r)
        return columns, results

    def __rich_repr__(self):
        yield "all_modules", self.all_modules
        yield "base_pipelines", self.base_pipelines

    def run(self, column_source):
        pass


@dataclass
class ColumnView:
    _events: Any
    _column_provenance: dict[Column, ModuleProvenance]
    _current_provenance: ModuleProvenance | None = None
    _allowed_inputs: ColumnCollection | None = None
    _allowed_outputs: ColumnCollection | None = None
    _allow_filter: bool = True
    metadata: Any | None = None
    pipeline_data: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True)
    class Key:
        metadata: Any
        pipeline_data: dict[str, Any]
        provenance: frozenset[tuple[Column, ModuleProvenance]]

    def copy(self):
        return ColumnView(
            _events=copy.copy(self._events),
            _column_provenance=copy.copy(self._column_provenance),
            metadata=copy.copy(self.metadata),
            pipeline_data=copy.copy(self.pipeline_data),
        )

    @staticmethod
    def fromEvents(events, provenance):
        return ColumnView(
            _events=events,
            _column_provenance={x: provenance for x in getAllColumns(events)},
            _current_provenance=provenance,
        )

    def getKeyForColumns(self, columns):
        ret = []
        for column in columns:
            for c, v in self._column_provenance.items():
                if column.contains(c):
                    ret.append((c, v))
        return hash(
            ColumnView.Key(
                freeze(self.metadata), freeze(self.pipeline_data), freeze(ret)
            )
        )

    def setColumn(self, events, column, value):
        column = Column(column)
        if len(column) == 1:
            return ak.with_field(events, value, column.fields)
        all_columns = list(self._column_provenance)
        max_col = max((column.commonParent(x) for x in all_columns), key=len)
        common = column[: len(max_col) + 1]
        rest = column[len(max_col) + 1 :]
        for c in reversed(list(rest.iterParts())):
            value = ak.zip({c: value})
        return ak.with_field(events, value, common.fields)

    def __setitem__(self, column, value):
        column = Column(column)
        if self._allowed_outputs is not None and not self._allowed_outputs.contains(
            column
        ):
            raise RuntimeError(
                f"Column {column} is not in the list of outputs {self._allowed_outputs}"
            )
        self._events = self.setColumn(self._events, column, value)
        all_columns = getAllColumns(self._events)
        self._column_provenance = {
            x: y for x, y in self._column_provenance.items() if x in all_columns
        }
        logger.info(
            f"Adding column {column} to events with provenance {self._current_provenance}"
        )
        for c in all_columns:
            if column.contains(c):
                self._column_provenance[c] = self._current_provenance

    def setColumnProvenance(self, column, provenance):
        columns = Column(column)
        all_columns = getAllColumns(self._events)
        for c in all_columns:
            if column.contains(c):
                self._column_provenance[c] = provenance

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
            with self.useProvenance(other.getProvenance(column)[column]):
                self[column] = other[column]

    def withColumnFrom(self, other, columns):
        new = copy.copy(self)
        new.addColumnsFrom(other, columns)
        return new

    def filter(self, mask):
        if not self._allow_filter:
            raise RuntimeError()
        new_cols = copy.copy(self)
        new_cols._events = new_cols._events[mask]
        for c in self._column_provenance:
            self._column_provenance[c] = self._current_provenance
        return new_cols

    @contextlib.contextmanager
    def useProvenance(self, provenance: ModuleProvenance):
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


@dataclass
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
        return columns, {}

    def getInputs(self):
        return [self.input_col]


@dataclass
class BQuarkMaker(AnalyzerModule):
    input_jets: Column
    output_col: Column
    working_point: float = 0.5

    def run(self, columns, params):
        disc = columns[self.input_jets].btagDeepFlavB
        bjets = columns[self.input_jets][disc > self.working_point]
        columns[self.output_col] = bjets
        return columns, {}

    def getInputs(self):
        return [self.input_jets]


@dataclass
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

        return columns, {}

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

    def getInputs(self):
        return [self.input_jets]

@dataclass
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

        return columns, {}

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

    def getInputs(self):
        return [self.input_jets]

@dataclass
class SomeShapeScaleFactor(AnalyzerModule):
    input_jets: Column
    output_jets: Column

    def run(self, columns, params):
        variation = params["variation"]
        return columns, {}

    def getParameterSpec(self):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=["central", "shape_up_1", "shape_down_1", "shape_up_2", "shape_down_2"],
                    tags={
                        "shape_variation",
                    },
                )
            }
        )

    def getInputs(self):
        return [self.input_jets]


@dataclass
class NObjFilter(AnalyzerModule):
    out_name: str
    input_col: str
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
        columns["Selection", self.out_name] = sel
        return columns, {}

    def getInputs(self):
        return [self.input_col]


@dataclass
class LoadColumns(AnalyzerModule):
    def getInputs(self):
        return []

    def getParameterSpec(self):
        return ModuleParameterSpec({"chunk": ParameterSpec()})

    def run(self, columns, params):
        fname = "nano_dy.root"
        events = NanoEventsFactory.from_root(
            {fname: "Events"},
            schemaclass=NanoAODSchema,
        ).events()
        ret = ColumnView.fromEvents(
            events,
            self.getProvenance(None, params),
        )
        return ret, []


def getPipelineSpecs(pipeline):
    ret = {}
    for node in pipeline:
        ret[node.node_id] = node.getModuleSpec()
    return AnalyzerParameterSpec(ret)


@dataclass
class HLTPass(AnalyzerModule):
    out_name: str
    hlt_trigger_names: list[str]

    def run(self, columns, params):
        ret = columns["HLT"][self.hlt_trigger_names[0]]
        for name in self.hlt_trigger_names[1:]:
            ret &= columns["HLT"][name]
        columns["Selection", self.out_name] = ret
        return columns, {}

    def init(self, config):
        pass

    def getInputs(self):
        return [Column("HLT")]


@dataclass
class SelectOnColumns(AnalyzerModule):
    column_names: list[Column]

    def run(self, columns, params):
        cols = self.getInputs()
        ret = columns[cols[0]]
        for name in cols[1:]:
            ret = ret & columns[name]
        columns = columns.filter(ret)
        return columns, {}

    def init(self, config):
        pass

    def getInputs(self):
        return [Column("Selection") + x for x in self.column_names]


@dataclass
class HistogramBuilder(AnalyzerModule):
    product_name: str
    columns: list[Column]
    axes: list[Any]

    def run(self, columns_set, params):
        breakpoint()
        return None, []

    def getInputs(self):
        return [*self.columns, Column("Categories")]


def makeHistogram(columns, params, name, axes, data, description, want_variations=None):
    if not isinstance(data, (list, tuple)):
        data = [data]
        axes = [axes]

    names = []
    for i, d in enumerate(data):
        name = Column(f"auto-col-{name}-{i}")
        names.append(name)
        columns[name] = d

    b = HistogramBuilder(names, names, axes)
    return ModuleAddition(
        b, [params.all_parameters], b.getParameterSpec().getWithValues({})
    )


@dataclass
class TestHistogram(AnalyzerModule):
    col: Column

    def run(self, columns, params):
        j = columns[self.col]
        h = makeHistogram(columns, params, "pt_1", None, j[:, 0].pt, None)
        return columns, [1, h]

    def init(self, config):
        pass

    def getInputs(self):
        return [self.col]


# class HTSelection(AnalyzerModule):
#     def getInputs(self):
#         return {"good_jets"}
#
#     def getProducts(self):
#         return {"HT"}
#
#     def run(self, columns, analyzer, **kwargs):
#         pass
#
#
# class GoodJets(AnalyzerModule):
#     def getInputs(self):
#         return {"Jet"}
#
#     def getProducts(self):
#         return {"good_jets"}
#
#     def run(self, columns, analyzer, **kwargs):
#         pass
#
#
# class JetSysts(AnalyzerModule):
#     def getInputs(self):
#         return {"Jet"}
#
#     def getProducts(self):
#         return {"CorrectedJet"}
#
#     def run(self, columns, analyzer, **kwargs):
#         pass
#
#
# class PuReweight(AnalyzerModule):
#     def getInputs(self):
#         return {}
#
#     def getProducts(self):
#         return {"Weights.PU_Weight"}
#
#     def run(self, columns, analyzer, **kwargs):
#         pass
#
#
# class HTHistogram(AnalyzerModule):
#     def getInputs(self):
#         return {"Weights", "HT"}
#
#     def run(self, columns, analyzer, **kwargs):
#         pass
#

#
#
# def main():
#     print("HERE")
#     pass
#
#
if __name__ == "__main__":
    load = LoadColumns()
    hlt_pass = HLTPass("pass_trigger", ["PFHT1050"])
    good_jets = GoodJetMaker(Column("Jet"), Column("GoodJet"))
    b_jets = BQuarkMaker(Column("GoodJet"), Column("MedBJet"))
    b_jets2 = BQuarkMaker(Column("GoodJet"), Column("TightBJet"))
    n_obj_filter = NObjFilter("med_b_jets", Column("MedBJet"), None, 1)
    trigger_filter = SelectOnColumns(["pass_trigger", "med_b_jets"])
    smaller_filter = SelectOnColumns(["pass_trigger"])
    some_sf = SomeScaleFactor(Column("GoodJet"))
    test_h = TestHistogram(Column("GoodJet"))

    pipeline = [
        load,
        hlt_pass,
        good_jets,
        b_jets,
        b_jets,
        b_jets2,
        n_obj_filter,
        SomeShapeScaleFactor(Column("GoodJet"), Column("CorrectedJets")),
        trigger_filter,
        some_sf,
        SomeOtherScaleFactor(Column("CorrectedJets")),
        SomeOtherScaleFactor(Column("CorrectedJets")),
        test_h,
    ]
    only_trigger = [
        load,
        hlt_pass,
        good_jets,
        b_jets,
        n_obj_filter,
        smaller_filter,
        some_sf,
        test_h,
    ]

    analyzer = Analyzer()
    analyzer.addPipeline("pipe1", pipeline)
    analyzer.addPipeline("pipe2", only_trigger)
    analyzer.initModules(None)
    p1 = analyzer.base_pipelines["pipe1"]
    spec = getPipelineSpecs(p1)
    vals = spec.getWithValues({"LoadColumns": {"chunk": "nano_dy.root"}})
    print(spec)
    print(spec.getTags("weight_variation"))
    # analyzer.runPipelineWithParameters(None, p1, vals)
    #
    # p1 = analyzer.base_pipelines["pipe2"]
    # spec = getPipelineSpecs(p1)
    # vals = spec.getWithValues({"LoadColumns-1": {"chunk": "nano_dy.root"}})
    # analyzer.runPipelineWithParameters(None, p1, vals)
