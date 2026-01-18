from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
import awkward as ak
from attrs import define




@define
class VecDRSelection(AnalyzerModule):
    input_col: Column
    selection_name: str
    min_dr: float | None = None
    max_dr: float | None = None
    idx_1: int = 0
    idx_2: int = 1

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]

    def run(self, columns, params):
        col = columns[self.input_col]
        filled = ak.pad_none(col, max([self.idx_1, self.idx_2]) + 1, axis=1)
        dr = ak.fill_none(filled[:, self.idx_1].delta_r(filled[:, self.idx_2]), False)
        sel = None
        if self.min_dr is not None:
            sel = dr >= self.min_dr
        if self.max_dr is not None:
            if sel is not None:
                sel = sel & (dr <= self.max_dr)
            else:
                sel = dr <= self.max_dr

        addSelection(columns, self.selection_name, sel)
        return columns, []


@define
class VecPt(AnalyzerModule):
    selection_name: str
    input_col: Column
    idx: int = 0
    min_pt: float | None = None
    max_pt: float | None = None

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]

    def run(self, columns, params):
        col = columns[self.input_col]
        filled = ak.pad_none(col, self.idx + 1, axis=1)
        pt = filled[:, self.idx].pt
        sel = None
        if self.min_pt is not None:
            sel = pt >= self.min_pt
        if self.max_pt is not None:
            if sel is not None:
                sel = sel & (pt <= self.max_pt)
            else:
                sel = pt <= self.max_pt

        sel = ak.fill_none(sel, False)
        addSelection(columns, self.selection_name, sel)
        return columns, []

