from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr
from analyzer.core.columns import Column
import operator as op
from attrs import define
from analyzer.core.columns import addSelection
import functools as ft


@define
class SimpleHLT(AnalyzerModule):
    """
    Select events based on HLT triggers.

    Parameters
    ----------
    triggers : list[str]
        List of trigger names to select.
    selection_name : str
        Name of the selection to be added to the columns.
    """

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
