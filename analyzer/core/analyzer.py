from __future__ import annotations

from attrs import define, field

import itertools as it

from collections import deque


from analyzer.core.serialization import converter
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.results import AnalyzerResult
from analyzer.modules.universal.load_columns import LoadColumns


def getPipelineSpecs(pipeline, metadata):
    ret = {}
    for node in pipeline:
        ret[node.node_id] = node.getModuleSpec(metadata)
    return PipelineParameterSpec(ret)

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
            current_spec = getPipelineSpecs(complete_pipeline, columns.metadata)
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
                    result_container.addResult(AnalyzerResult(res))
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
