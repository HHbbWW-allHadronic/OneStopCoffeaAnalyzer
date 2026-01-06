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
    """
    Select electrons based on kinematics, cut-based ID, and isolation.

    This analyzer filters electrons in an event according to minimum
    transverse momentum, maximum pseudorapidity, cut-based ID working point,
    and maximum mini-isolation.

    Parameters
    ----------
    input_col : Column
        Column containing the input electron collection.
    output_col : Column
        Column where the selected electrons will be stored.
    working_point : CutBasedWPs
        Cut-based ID working point (fail, veto, loose, medium, tight).
    min_pt : float, optional
        Minimum transverse momentum in GeV, by default 10.
    max_abs_eta : float, optional
        Maximum absolute pseudorapidity, by default 2.4.
    max_mini_iso : float, optional
        Maximum mini-isolation, by default 0.1.

    """
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
