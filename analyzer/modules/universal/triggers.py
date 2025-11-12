from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram


@define
class HLTFilter(AnalyzerModule):
    selection_name: str
    trigger_names: list[str] 

    def run(self, columns, params):
        era_trigger_names = columns.metadata["era"]["trigger_names"]
        pass_any = ft.reduce(op.or_, (columns["HLT"][era_trigger_names[x]] for x in self.trigger_names))
        columns["Selection", self.selection_col] = pass_any
        return columns, []

    def inputs(self, metadata):
        return [Column("HLT")]

    def outputs(self, metadata):
        return [self.selection_col]
