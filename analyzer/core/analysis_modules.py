from __future__ import annotations
import functools as ft
from cattrs.strategies import include_subclasses, configure_tagged_union



from attrs import define, field, make_class
from attrs import define, field
from analyzer.utils.structure_tools import freeze

from collections.abc import Collection

import contextlib

import abc
from typing import Any


@define(frozen=True)
class ModuleParameterValues:
    param_values: frozenset[tuple[str, Any]] = field(converter=freeze)
    spec: ModuleParameterSpec | None = field(eq=False, repr=False, default=None)

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
    metadata: dict[str, Any] = field(factory=dict)
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

    def getParameterSpec(self) -> ModuleParameterSpec:
        return ModuleParameterSpec()

    def preloadForMeta(self, metadata):
        pass

    def getColumnKey(self, columns):
        if isinstance(columns, ColumnView):
            return columns.getKeyForColumns(self.inputs(columns.metadata))
        else:
            return frozenset(self.getColumnKey(x) for x in (columns or []))

    def neededResources(self, metadata):
        return []

    def getKey(self, columns, params):
        ret = hash((self.name(), freeze(params), self.getColumnKey(columns)))
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
        columns = [(x, y.copy()) for x, y in columns]
        just_cols = [x[1] for x in columns]
        key = self.getKey(just_cols, params)
        logger.info(f"Execution key is {key}")
        logger.info(f"Cached keys are {list(self.__cache)}")
        if key in self.__cache:
            logger.info(f"Found key, using cached result")
            ret = self.__cache[key]
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


def configureConverter(conv):
    union_strategy = ft.partial(configure_tagged_union, tag_name="module_name")
    include_subclasses(AnalyzerModule, conv, union_strategy=union_strategy)
