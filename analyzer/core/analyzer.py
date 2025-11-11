from __future__ import annotations
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
from analyzer.core.serialization import converter
from analyzer.core.analyzer_modules import AnalyzerModule
from analyzer.core.columns import ColumnView
from analyzer.core.results import WrappedResult


def getPipelineSpecs(pipeline):
    ret = {}
    for node in pipeline:
        ret[node.node_id] = node.getModuleSpec()
    return PipelineParameterSpec(ret)


class Analyzer:
    def __init__(self):
        self.all_modules = []
        self.base_pipelines: dict[str, list[Node]] = {}

    def initModules(self, metadata):
        for m in self.all_modules:
            m.preloadForMeta(metadata)

    def getUniqueNode(self, pipeline, module):
        base = module.name()
        to_use, i = base, 0
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
        ret.append(self.getUniqueNode(ret, LoadColumns()))
        for module in pipeline:
            ret.append(self.getUniqueNode(ret, module))
        self.base_pipelines[name] = ret

    def runPipelineWithParameters(
        self,
        columns,
        pipeline,
        params,
        pipeline_meta=None,
        freeze_pipeline=False,
        handle_results=True,
    ):
        complete_pipeline = []
        column_metadata = columns.metadata
        pipeline_metadata = pipeline_meta or {}
        to_add = deque(pipeline)
        if handle_results:
            result_container = ResultContainer()
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
                    result_container.addResult(
                        WrappedResult(column_metadata, pipeline_metadata, res)
                    )
                elif isinstance(res, ModuleAddition) and not freeze_pipeline:
                    module = res.analyzer_module
                    if res.run_builder is None:
                        logger.info(f"Adding new module {module} to pipeline")
                        to_add.appendleft(self.getUniqueNode(complete_pipeline, module))
                    else:
                        logger.info(f"Running multi-parameter pipeline")

                        param_dicts = res.run_builder(current_spec, columns.metadata)
                        to_run = [params.withNewValues(x) for x in param_dicts]

                        everything = []
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

        result_container = ResultContainer("ChunkResult")

        for k, pipeline in self.base_pipelines.items():
            if k not in pipelines:
                continue
            spec = getPipelineSpecs(pipeline)
            vals = spec.getWithValues({"chunk": chunk})
            _, ret = self.runPipelineWithParameters(
                None, pipeline, vals, pipeline_name=k
            )
            result_container.addResult(ret)
        return result_container


module_unstr = converter.get_unstructure_hook(AnalyzerModule)
module_str = converter.get_structure_hook(AnalyzerModule)


@converter.register_structure_hook
def deserialize(data, t) -> Analyzer:
    analyzer = t()
    for k, v in data.items():
        analyzer.addPipeline(k, converter.structure(v, list[AnalyzerModule]))
    return analyzer


@converter.register_unstructure_hook
def serialize(analyzer: Analyzer):
    return {
        x: converter.unstructure([z.analyzer_module for z in y], list[AnalyzerModule])
        for x, y in analyzer.base_pipelines.items()
    }
