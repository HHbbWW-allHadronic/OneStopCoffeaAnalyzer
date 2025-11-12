from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define

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
