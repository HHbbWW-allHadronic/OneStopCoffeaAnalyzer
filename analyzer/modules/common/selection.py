from analyzer.core.analysis_modules import AnalyzerModule, register_module
import awkward as ak
from analyzer.core.columns import Column, EVENTS
from attrs import define, field
from analyzer.core.results import SelectionFlow
from .axis import RegularAxis
from .histogram_builder import makeHistogram

EVENTS = "EVENTS"


@define
class SelectOnColumns(AnalyzerModule):
    sel_name: str
    selection_names: list[str] = None
    save_cutflow: bool = True

    def run(self, columns, params):
        if self.selection_names is not None:
            cuts = self.selection_names
        else:
            cuts = [
                x
                for x, y in columns.pipeline_data.get("Selections", {}).items()
                if not y
            ]

        initial = ak.num(columns._events, axis=0)

        ret = columns[Column("Selection") + cuts[0]]
        cutflow = {"inital": initial, cuts[0]: ak.count_nonzero(ret, axis=0)}

        for name in cuts[1:]:
            ret = ret & columns[name]
            cutflow[name] = ak.count_nonzero(ret, axis=0)

        columns = columns.filter(ret)
        return columns, [
            SelectionFlow(self.sel_name, cuts=self.selection_names, cutflow=cutflow)
        ]

    def inputs(self, metadata):
        if self.selection_names is None:
            return [Column(("Selection"))]
        else:
            return [Column("Selection") + x for x in self.selection_names]

    def outputs(self, metadata):
        return EVENTS


@define
class NObjFilter(AnalyzerModule):
    selection_name: str
    input_col: Column
    min_count: int | None = None
    max_count: int | None = None

    def run(self, columns, analyzer, **kwargs):
        objs = columns[self.input_col]
        count = ak.num(objs, axis=1)
        sel = None
        if self.min_count is not None:
            sel = count >= self.min_count
        if self.max_count is not None:
            if sel is not None:
                sel = sel & (count <= self.max_count)
            else:
                sel = count <= self.max_count
        columns["Selection", self.selection_name] = sel
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]
