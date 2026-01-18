from __future__ import annotations
from typing import Literal
from analyzer.core.param_specs import (
    ModuleParameterSpec,
    ModuleParameterValues,
)
import functools as ft
from cattrs.strategies import include_subclasses, configure_tagged_union
from analyzer.core.run_builders import RunBuilder, DEFAULT_RUN_BUILDER
from attrs import define, field, make_class
from analyzer.core.results import ResultBase
from analyzer.utils.structure_tools import freeze, SimpleCache
from analyzer.core.columns import TrackedColumns, Column, ColumnCollection
import contextlib
import abc
from typing import Any
import logging

logger = logging.getLogger("analyzer.core")


@define
class MetadataExpr(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, metadata) -> bool: ...


@define
class IsYear(MetadataExpr):
    year: str

    def evaluate(self, metadata):
        return metadata["era"]["name"] == self.year


@define
class IsSampleType(MetadataExpr):
    sample_type: str

    def evaluate(self, metadata):
        return metadata["sample_type"] == self.sample_type


@define
class IsRun(MetadataExpr):
    run: int

    def evaluate(self, metadata):
        is_run_2 = any(x in metadata["era"]["name"] for x in ("2016", "2017", "2018"))
        return (self.run == 2) == is_run_2


@define
class MetadataAnd(MetadataExpr):
    require_all: list[MetadataExpr]

    def evaluate(self, metadata):
        return all(x.evaluate(metadata) for x in self.require_all)


@define
class MetadataOr(MetadataExpr):
    require_any: list[MetadataExpr]

    def evaluate(self, metadata):
        return any(x.evaluate(metadata) for x in self.require_any)


@define
class MetadataNot(MetadataExpr):
    require_not: MetadataExpr

    def evaluate(self, metadata):
        return not self.require_not.evaluate(metadata)


MultiColumns = list[tuple[Any, TrackedColumns]]


@define
class BaseAnalyzerModule(abc.ABC):
    MAX_CACHE_SIZE = 25

    _cache: SimpleCache = field(
        factory=lambda: SimpleCache(max_size=AnalyzerModule.MAX_CACHE_SIZE),
        init=False,
        repr=False,
    )
    should_run: MetadataExpr | None = field(default=None, kw_only=True)

    @abc.abstractmethod
    def inputs(
        self, metadata
    ) -> ColumnCollection | list[Column] | Literal["EVENTS"]: ...

    def getParameterSpec(self, metadata: dict) -> ModuleParameterSpec:
        return ModuleParameterSpec()

    def preloadForMeta(self, metadata: dict):
        pass

    def getFromCache(self, key):
        return self._cache[key]

    def neededResources(self, metadata) -> list[str]:
        return []

    @classmethod
    def name(cls):
        return cls.__name__

    def clearCache(self):
        self._cache.clear()


@define
class AnalyzerModule(BaseAnalyzerModule):
    @abc.abstractmethod
    def outputs(
        self, metadata
    ) -> ColumnCollection | list[Column] | Literal["EVENTS"]: ...

    @abc.abstractmethod
    def run(
        self, columns, params: ModuleParameterValues
    ) -> tuple[TrackedColumns, list[ResultBase | ModuleAddition]]:
        pass

    def getKey(self, columns, params):
        inp = self.inputs(columns.metadata)
        if inp == "EVENTS":
            k = columns.getKeyForAll()
        else:
            k = columns.getKeyForColumns(self.inputs(columns.metadata))
        ret = hash((self.name(), freeze(params), k))
        return ret

    def getKeyNoParams(self, columns):
        inp = self.inputs(columns.metadata)
        if inp == "EVENTS":
            k = columns.getKeyForAll()
        else:
            k = columns.getKeyForColumns(self.inputs(columns.metadata))
        ret = hash((self.name(), k))
        return ret

    def __run(self, columns, params):
        orig_columns = columns
        columns = columns.copy()
        key = self.getKey(columns, params)
        logger.debug(f"Execution key is {key}")
        logger.debug(f"Cached keys are {list(self._cache)}")
        if key in self._cache:
            logger.debug("Found key, using cached result")
            cached_cols, r, internal = self._cache[key]
            outputs = self.outputs(columns.metadata)
            if outputs == "EVENTS":
                return cached_cols, r
            outputs += internal
            columns.addColumnsFrom(cached_cols, outputs)
            columns.pipeline_data = cached_cols.pipeline_data
            return columns, r
        logger.debug(f"Did not find cached result, running module {self.name}")
        outputs = self.outputs(columns.metadata)
        inputs = self.inputs(columns.metadata)
        if outputs == "EVENTS":
            output_cx = columns.allowedOutputs(outputs)
        else:
            output_cx = contextlib.nullcontext()

        if inputs == "EVENTS":
            inputs_cx = columns.allowedInputs(outputs)
        else:
            inputs_cx = contextlib.nullcontext()

        with (
            columns.useKey(key),
            inputs_cx,
            output_cx,
        ):
            _, res = self.run(columns, params)
            internal = columns.updatedColumns(orig_columns, Column("INTERNAL_USE"))
        self._cache[key] = (columns, res, internal)
        return columns, res

    def __call__(self, columns, params):
        try:
            logger.debug(f"Running analyzer module {self}")
            if self.should_run is not None and columns is not None:
                metadata = columns.metadata
                should_run = self.should_run.evaluate(metadata)
                if not should_run:
                    return columns, []
            return self.__run(columns, params)
        except Exception as e:
            logger.error(f"An exception occurred while running module {self}")
            raise e


@define
class EventSourceModule(BaseAnalyzerModule):
    @abc.abstractmethod
    def outputs(self, metadata) -> ColumnCollection | list[Column]: ...

    @abc.abstractmethod
    def run(self, params: ModuleParameterValues) -> TrackedColumns:
        pass

    def getKey(self, params):
        ret = hash((self.name(), freeze(params)))
        return ret

    def __call__(self, params):
        try:
            logger.debug(f"Running analyzer module {self}")
            key = self.getKey(params)
            logger.debug(f"Execution key is {key}")
            logger.debug(f"Cached keys are {list(self._cache)}")
            if key in self._cache:
                logger.debug("Found key, using cached result")
                cached_cols = self._cache[key]
                return cached_cols
            logger.debug(f"Did not find cached result, running module {self}")
            ret = self.run(params)
            self._cache[key] = ret
            return ret
        except Exception as e:
            logger.error(f"An exception occurred while running module {self}")
            raise e


@define
class PureResultModule(BaseAnalyzerModule):
    @abc.abstractmethod
    def outputs(self, metadata) -> ColumnCollection | list[Column]: ...

    @abc.abstractmethod
    def run(
        self, columns: MultiColumns, params: ModuleParameterValues
    ) -> list[ResultBase]:
        pass

    def getKey(self, columns: MultiColumns, params: ModuleParameterValues):
        ret = hash(
            (
                self.name(),
                freeze(params),
                frozenset(
                    (x, y.getKeyForColumns(self.inputs(y.metadata)))
                    for x, y in (columns or [])
                ),
            )
        )
        return ret

    def __run(self, columns: MultiColumns, params):
        columns = [(x, y.copy()) for x, y in columns]
        just_cols = [x[1] for x in columns]
        key = self.getKey(columns, params)
        logger.debug(f"Execution key is {key}")
        logger.debug(f"Cached keys are {list(self._cache)}")
        if key in self._cache:
            logger.debug("Found key, using cached result")
            ret = self._cache[key]
            return ret
        logger.debug(f"Did not find cached result, running module {self.name}")
        with contextlib.ExitStack() as stack:
            for c in just_cols:
                stack.enter_context(c.useKey(key))
                stack.enter_context(c.allowedOutputs(self.outputs(c.metadata)))
                stack.enter_context(c.allowedInputs(self.inputs(c.metadata)))
            ret = self.run(columns, params)
        self._cache[key] = ret
        return ret

    def __call__(self, columns: MultiColumns, params):
        try:
            logger.debug(f"Running analyzer module {self}")
            # if self.should_run is not None and columns is not None:
            #     metadata = columns.metadata
            #     should_run = self.should_run.evaluate(metadata)
            #     if not should_run:
            #         return columns, []
            return self.__run(columns, params)
        except Exception as e:
            logger.error(f"An exception occurred while running module {self}")
            raise e


def defaultCols(columns):
    def inner(self, metadata):
        return [Column(x) for x in columns]

    return inner


def defaultParameterSpec(params):
    def inner(self, metadata):
        return ModuleParameterSpec(params)

    return inner


def register_module(input_columns, output_columns, configuration=None, params=None):
    configuration = configuration or {}
    params = params or {}

    def wrapper(func):
        getParameterSpec = defaultParameterSpec(params)
        run = func
        if callable(input_columns):
            inputs = input_columns
        else:
            inputs = defaultCols(input_columns)

        if callable(output_columns):
            outputs = output_columns
        else:
            outputs = defaultCols(output_columns)

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
class ModuleAddition:
    analyzer_module: AnalyzerModule | PureResultModule
    run_builder: RunBuilder | type | None = DEFAULT_RUN_BUILDER
    this_module_parameters: dict | None = None
    # parameter_runs: list[PipelineParameterValues]


def configureConverter(conv):
    union_strategy = ft.partial(configure_tagged_union, tag_name="module_name")
    include_subclasses(AnalyzerModule, conv, union_strategy=union_strategy)
    include_subclasses(MetadataExpr, conv)
