from __future__ import annotations
import copy


from attrs import define, field, asdict

import itertools as it

from collections import deque

from analyzer.core.analysis_modules import (
    AnalyzerModule,
    ModuleAddition,
    PureResultModule,
    EventSourceModule,
)
from analyzer.core.param_specs import PipelineParameterSpec
from analyzer.core.run_builders import DEFAULT_RUN_BUILDER, CompleteSysts, RunBuilder
from analyzer.core.results import (
    ResultProvenance,
    ResultGroup,
    ResultBase,
)
from collections import ChainMap
from analyzer.modules.common.load_columns import LoadColumns
import logging

from analyzer.utils.structure_tools import SimpleCache, freeze, flatten


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
    analyzer_module: EventSourceModule | AnalyzerModule | PureResultModule

    def getModuleSpec(self, metadata):
        return self.analyzer_module.getParameterSpec(metadata)

    def __eq__(self, other):
        return (
            self.node_id == other.node_id
            and self.analyzer_module == other.analyzer_module
        )

    def __call__(self, columns, params):
        params = params[self.node_id]
        if isinstance(self.analyzer_module, EventSourceModule):
            return self.analyzer_module(params)
        else:
            return self.analyzer_module(columns, params)

    def __rich_repr__(self):
        yield "node_id", self.node_id
        yield "module_id", id(self.analyzer_module)
        yield "analyzer_module", self.analyzer_module


@define
class Analyzer:
    all_modules: list = field(factory=list)
    base_pipelines: dict[str, list[Node]] = field(factory=dict)

    default_run_builder: RunBuilder = field(factory=CompleteSysts)

    _cache: SimpleCache = field(factory=SimpleCache)

    def initModules(self, metadata):
        pass
        # for m in self.all_modules:
        #     m.preloadForMeta(metadata)

    def clearCaches(self):
        self._cache.clear()
        for m in self.all_modules:
            m.clearCache()

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
        self, pipeline, params, freeze_pipeline=False, result_container_name=None
    ):
        node_ids = [n.node_id for n in pipeline]
        params = {x: y for x, y in params.items() if x in node_ids}
        key = hash(freeze((node_ids, params)))

        logger.debug(f"Pipeline execution key is {key}")
        if key in self._cache:
            logger.debug(f"Found key {key}, using cached columns")
            return self._cache[key], None
        else:
            logger.debug(f"Did not find key {key}, recomputing")
        params = copy.deepcopy(params)
        complete_pipeline = []
        to_add = deque(pipeline)
        current_spec, columns = None, None

        if result_container_name is None:
            result_container = None
        else:
            result_container = ResultGroup(
                result_container_name, metadata={"pipeline": result_container_name}
            )

        while to_add:
            head = to_add.popleft()
            complete_pipeline.append(head)
            if columns is not None:
                columns = columns.copy()
                current_spec = getPipelineSpecs(complete_pipeline, columns.metadata)
                columns, results = head(columns, params)
            else:
                columns, results = head(columns, params), []

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
                        node = self.getUniqueNode(complete_pipeline, module)
                        complete_pipeline.append(node)
                        logger.debug(f"New node id is {node.node_id}")
                        params = ChainMap(
                            params, {node.node_id: res.this_module_parameters}
                        )
                    else:
                        logger.debug("Running multi-parameter pipeline")
                        if res.run_builder is DEFAULT_RUN_BUILDER:
                            run_builder = self.default_run_builder
                        else:
                            run_builder = res.run_builder

                        param_dicts = run_builder(current_spec, columns.metadata)
                        to_run = [
                            (x, current_spec.getWithValues(params, y))
                            for x, y in param_dicts
                        ]
                        everything = []
                        for name, params_set in to_run:
                            c, _ = self.runPipelineWithParameters(
                                complete_pipeline,
                                params_set,
                                freeze_pipeline=True,
                                result_container_name=None,
                            )
                            everything.append((name, c))
                        logger.debug(
                            f"Running node {module} with {len(everything)} parameter sets"
                        )
                        r = module(everything, res.this_module_parameters or {})
                        results.extendleft(r)
                else:
                    raise RuntimeError(
                        f"Invalid object type returned from analyzer module. {res}"
                    )
        self._cache[key] = columns
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
        metadata = copy.deepcopy(metadata)
        metadata["chunk"] = asdict(chunk)

        for k, pipeline in self.base_pipelines.items():
            if k not in pipelines:
                continue
            spec = getPipelineSpecs(pipeline, metadata)
            vals = spec.getWithValues(
                {"ENTRYPOINT": {"chunk": chunk, "metadata": metadata}}
            )
            _, result = self.runPipelineWithParameters(
                pipeline,
                vals,
                result_container_name=k,
            )
            pipeline_container.addResult(result)
        return root_container

    @classmethod
    def _structure(cls, data: dict, conv) -> Analyzer:
        analyzer = cls()
        builder = data.pop("default_run_builder", None)
        if builder is not None:
            analyzer.default_run_builder = conv.structure(builder, RunBuilder)

        for k, v in data.items():
            analyzer.addPipeline(
                k, [conv.structure(x, AnalyzerModule) for x in flatten(v)]
            )
        return analyzer

    def _unstructure(self, conv) -> dict:
        return {
            x: conv.unstructure([z.analyzer_module for z in y])
            for x, y in self.base_pipelines.items()
        }
