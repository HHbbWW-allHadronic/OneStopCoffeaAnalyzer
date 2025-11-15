from analyzer.core.analysis_modules import (
    AnalyzerModule,
    ModuleParameterSpec,
    ParameterSpec,
)
from analyzer.core.columns import Column, ColumnView
from analyzer.core.event_collection import FileChunk
from attrs import define


@define
class LoadColumns(AnalyzerModule):
    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return None

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "chunk": ParameterSpec(param_type=FileChunk, tags={"column-input"}),
                "metadata": ParameterSpec(param_type=dict, tags={"column-metadata"}),
            }
        )

    def run(self, columns, params):
        key = self.getKey(None, params)
        ret = params["chunk"].loadEvents(
            "coffea-virtual",
            view_kwargs=dict(metadata=params["metadata"], provenance=key),
        )
        return ret, []
