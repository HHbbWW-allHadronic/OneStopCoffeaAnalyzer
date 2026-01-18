from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
import operator as op
from attrs import define
from analyzer.core.columns import addSelection
import functools as ft




@define
class SimpleHLT(AnalyzerModule):
    triggers: list[str]
    selection_name: str = "PassHLT"

    def run(self, columns, params):
        metadata = columns.metadata
        trigger_names = metadata["era"]["trigger_names"]
        hlt = columns["HLT"]
        pass_trigger = ft.reduce(
            op.or_, (hlt[trigger_names[name]] for name in self.triggers)
        )
        addSelection(columns, self.selection_name, pass_trigger)
        return columns, []

    def inputs(self, metadata):
        return [Column(("HLT"))]

    def outputs(self, metadata):
        return [Column(f"Selection.{self.selection_name}")]
