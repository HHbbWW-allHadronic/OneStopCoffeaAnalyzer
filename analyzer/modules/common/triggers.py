from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define


@define
class HLTFilter(AnalyzerModule):
    """
    Apply an event-level filter based on HLT info.
    Triggers are or-ed together.

    Parameters
    ----------
    selection_name : str
        Name of the selection column.
    trigger_names : list of str
        List of trigger names to check against the event HLT information.
    """

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
