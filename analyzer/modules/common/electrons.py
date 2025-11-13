from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
import correctionlib
import enum


class CutBasedWPs(enum.Enum):
    fail =  0
    veto = 1
    loose = 2
    medium = 3
    tight = 3

@define
class ElectronMaker(AnalyzerModule):
    input_col: Column
    output_col: Column
    working_point: CutBasedWPs
    min_pt: float = 10
    max_abs_eta: float = 2.4

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        electrons = columns[self.input_col]
        pass_pt = electrons.pt > self.min_pt
        pass_eta = abs(electrons.eta) < self.max_abs_eta
        pass_wp = electrons.cutBased >= self.working_point
        columns[self.output_col] = electrons[pass_pt & pass_eta & pass_wp]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
