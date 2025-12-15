from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
import correctionlib
import enum


class CutBasedWPs(str, enum.Enum):
    fail = "fail"
    veto = "veto"
    loose = "loose"
    medium = "medium"
    tight = "tight"


cut_mapping = dict(fail=0, veto=1, loose=2, medium=3, tight=4)


@define
class ElectronMaker(AnalyzerModule):
    input_col: Column
    output_col: Column
    working_point: CutBasedWPs
    min_pt: float = 10
    max_abs_eta: float = 2.4
    max_mini_iso: float = 0.1

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        electrons = columns[self.input_col]
        pass_pt = electrons.pt > self.min_pt
        pass_eta = abs(electrons.eta) < self.max_abs_eta
        pass_wp = electrons.cutBased >= cut_mapping[self.working_point]
        pass_iso = electrons.miniPFRelIso_all < self.max_mini_iso
        columns[self.output_col] = electrons[pass_pt & pass_eta & pass_wp & pass_iso]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
