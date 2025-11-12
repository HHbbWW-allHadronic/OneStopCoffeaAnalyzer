from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram

EVENTS = "EVENTS"

@define
class SelectOnColumns(AnalyzerModule):
    column_names: list[str]

    def run(self, columns, params):
        cols = self.inputs(columns.metadata)
        ret = columns[cols[0]]
        for name in cols[1:]:
            ret = ret & columns[name]
        columns = columns.filter(ret)
        return columns, []

    def inputs(self, metadata):
        return [Column("Selection") + x for x in self.column_names]

    def outputs(self, metadata):
        return EVENTS
