from __future__ import annotations
import copy
import cProfile

from rich import print
from attrs import define, field, frozen

import itertools as it

from collections import deque

from analyzer.core.columns import TrackedColumns
from analyzer.core.serialization import converter
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    PipelineParameterSpec,
    ModuleAddition,
)
from analyzer.core.results import (
    ResultProvenance,
    ResultGroup,
    ResultBase,
)
from collections import ChainMap
from analyzer.modules.common.load_columns import LoadColumns
import logging

from analyzer.utils.structure_tools import SimpleCache, freeze


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

    __cache: SimpleCache = field(factory=SimpleCache)

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

    def neededResources(self, metadata):
        needed_resources = []
        for module in self.all_modules:
            needed_resources.extend(module.neededResources(metadata))
        return needed_resources

    def addPipeline(self, name, pipeline):
        ret = []
        node = Node("ENTRYPOINT", LoadColumns())
        ret.append(node)
        for module in pipeline:
            ret.append(self.getUniqueNode(ret, module))
        self.base_pipelines[name] = ret

    def runPipelineWithParameters(
        self,
        columns,
        pipeline,
        params,
        freeze_pipeline=False,
        result_container=None,
    ):
        key = hash(freeze(([n.node_id for n in pipeline], params)))

        logger.debug(f"Pipeline execution key is {key}")
        if key in self.__cache:
            logger.debug(f"Found key {key}, using cached columns")
            return self.__cache[key], None

        params = copy.deepcopy(params)
        complete_pipeline = []
        to_add = deque(pipeline)
        current_spec = None

        while to_add:
            head = to_add.popleft()
            complete_pipeline.append(head)

            if columns is not None:
                columns = columns.copy()
                current_spec = getPipelineSpecs(complete_pipeline, columns.metadata)
                columns, results = head(columns, params)
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
                        logger.debug(f"Adding new module {module} to pipeline")
                        spec = module.getParameterSpec(columns.metadata)
                        node = self.getUniqueNode(complete_pipeline, module)
                        complete_pipeline.append(node)
                        logger.debug(f"New node id is {node.node_id}")
                        params = ChainMap(
                            params, {node.node_id: res.this_module_parameters}
                        )
                    else:
                        logger.debug(f"Running multi-parameter pipeline")

                        param_dicts = res.run_builder(current_spec, columns.metadata)
                        to_run = [
                            (x, current_spec.getWithValues(params, y))
                            for x, y in param_dicts
                        ]
                        everything = []
                        for name, params_set in to_run:
                            c, _ = self.runPipelineWithParameters(
                                None,
                                complete_pipeline,
                                params_set,
                                freeze_pipeline=True,
                                result_container=None,
                            )
                            everything.append((name, c))
                        logger.debug(
                            f"Running node {module} with {len(everything)} parameter sets"
                        )
                        r = module(everything, res.this_module_parameters or {})
                        results.extendleft(r)
                else:
                    raise RuntimeError(
                        f"Invalid object type returned from analyzer module."
                    )
        self.__cache[key] = columns
        return columns, result_container

    def run(self, chunk, metadata, pipelines=None):
        pipelines = pipelines or list(self.base_pipelines)

        root_container = ResultGroup("ROOT")
        dataset_container = ResultGroup(metadata["dataset_name"])
        sample_container = ResultGroup(metadata["sample_name"], metadata=metadata)
        pipeline_container = ResultGroup("pipelines")

        root_container.addResult(dataset_container)
        dataset_container.addResult(sample_container)
        sample_container.addResult(ResultProvenance("_provenance", chunk.toFileSet()))
        sample_container.addResult(pipeline_container)

        with cProfile.Profile() as pr:
            for k, pipeline in self.base_pipelines.items():
                if k not in pipelines:
                    continue
                pipeline_result = ResultGroup(k)
                spec = getPipelineSpecs(pipeline, metadata)
                vals = spec.getWithValues(
                    {"ENTRYPOINT": {"chunk": chunk, "metadata": metadata}}
                )
                self.runPipelineWithParameters(
                    None,
                    pipeline,
                    vals,
                    result_container=pipeline_result,
                )
                pipeline_container.addResult(pipeline_result)
            pr.dump_stats("test.prof")
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
