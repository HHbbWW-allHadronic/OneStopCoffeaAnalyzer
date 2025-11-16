from __future__ import annotations
import copy

from rich import print
from attrs import define, field

import itertools as it

from collections import deque

from analyzer.core.serialization import converter
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    PipelineParameterSpec,
    ModuleAddition,
)
from analyzer.core.results import (
    ResultProvenance,
    AnalyzerResult,
    ResultContainer,
    ResultBase,
)
from analyzer.modules.common.load_columns import LoadColumns
import logging


# Define the log message format
FORMAT = "%(message)s"

# Configure basic logging with RichHandler
logging.basicConfig(
    level="WARNING",  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    format=FORMAT,
)
logger = logging.getLogger("analyzer.core")


def getPipelineSpecs(pipeline, metadata):
    ret = {}
    for node in pipeline:
        ret[node.node_id] = node.getModuleSpec(metadata)
    return PipelineParameterSpec(ret)


@define
class Node:
    node_id: str
    analyzer_module: AnalyzerModule

    def getModuleSpec(self, metadata):
        return self.analyzer_module.getParameterSpec(metadata)

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
class Analyzer:
    all_modules: list = field(factory=list)
    base_pipelines: dict[str, list[Node]] = field(factory=dict)

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
        ret.append(Node("ENTRYPOINT", LoadColumns()))
        for module in pipeline:
            ret.append(self.getUniqueNode(ret, module))
        self.base_pipelines[name] = ret

    def runPipelineWithParameters(
        self,
        columns,
        pipeline,
        params,
        execution_name=None,
        pipeline_meta=None,
        freeze_pipeline=False,
        result_container=None,
    ):
        params = copy.deepcopy(params)
        complete_pipeline = []
        to_add = deque(pipeline)

        while to_add:
            head = to_add.popleft()
            complete_pipeline.append(head)

            if columns is not None:
                columns = columns.copy()
                current_spec = getPipelineSpecs(complete_pipeline, columns.metadata)
                columns, results = head(columns, params)
                pipeline_metadata = {"execution_name": execution_name}
                columns.pipeline_data = pipeline_metadata
            else:
                columns, results = head(columns, params)

            if not result_container:
                continue
            results = deque(results)

            while results:
                res = results.popleft()
                if isinstance(res, ResultBase):
                    result_container.addResult(res)
                elif isinstance(res, ModuleAddition) and not freeze_pipeline:
                    module = res.analyzer_module
                    if res.run_builder is None:
                        logger.info(f"Adding new module {module} to pipeline")
                        spec = module.getParameterSpec(columns.metadata)
                        node = self.getUniqueNode(complete_pipeline, module)
                        params.values[node.node_id] = spec.getWithValues(
                            res.this_module_parameters or {}
                        )
                        to_add.appendleft(node)
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
                                result_container=None,
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
                    breakpoint()
                    raise RuntimeError(
                        f"Invalid object type returned from analyzer module."
                    )
        return columns, result_container

    def run(self, chunk, metadata, pipelines=None):
        pipelines = pipelines or list(self.base_pipelines)

        root_container = ResultContainer("Dataset")
        dataset_container = ResultContainer(metadata["dataset_name"])
        sample_container = ResultContainer(metadata["sample_name"])
        pipeline_container = ResultContainer("pipelines")

        root_container.addResult(dataset_container)
        dataset_container.addResult(sample_container)
        sample_container.addResult(ResultProvenance("_provenance", chunk.toFileSet()))
        sample_container.addResult(pipeline_container)

        for k, pipeline in self.base_pipelines.items():
            if k not in pipelines:
                continue
            pipeline_result = ResultContainer(k)
            spec = getPipelineSpecs(pipeline, metadata)
            vals = spec.getWithValues(
                {"ENTRYPOINT": {"chunk": chunk, "metadata": metadata}}
            )
            _, ret = self.runPipelineWithParameters(
                None,
                pipeline,
                vals,
                result_container=pipeline_result,
                execution_name="central",
            )
            pipeline_container.addResult(pipeline_result)
        return root_container

    @classmethod
    def _structure(cls, data: dict, conv) -> Analyzer:
        analyzer = cls()
        for k, v in data.items():
            analyzer.addPipeline(k, [conv.structure(x, AnalyzerModule) for x in v])
        return analyzer

    def _unstructure(self, conv) -> dict:
        return {
            x: conv.unstructure([z.analyzer_module for z in y])
            for x, y in self.base_pipelines.items()
        }
