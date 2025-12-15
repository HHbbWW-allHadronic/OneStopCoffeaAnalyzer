from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
import enum
import correctionlib


class IdWps(str, enum.Enum):
    loose = "looseId"
    medium = "mediumId"
    tight = "tightId"


class IsoWps(str, enum.Enum):
    very_loose = "very_loose"
    loose = "loose"
    medium = "medium"
    tight = "tight"
    very_tight = "very_tight"
    very_very_tight = "very_very_tight"


cut_mapping = dict(
    very_loose=1, loose=2, medium=3, tight=4, very_tight=5, very_very_tight=6
)


@define
class MuonMaker(AnalyzerModule):
    input_col: Column
    output_col: Column
    id_working_point: IdWps
    min_pt: float = 10
    max_abs_eta: float = 2.4
    max_mini_iso: float = 0.1
    iso_working_point: IsoWps | None = None

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        muon = columns[self.input_col]
        pass_pt = muon.pt > self.min_pt
        pass_eta = abs(muon.eta) < self.max_abs_eta
        pass_id_wp = muon[self.id_working_point]
        pass_mini_iso = muon.miniPFRelIso_all < self.max_mini_iso
        passed = pass_pt & pass_eta & pass_id_wp & pass_mini_iso
        if self.iso_working_point is not None:
            pass_iso_wp = muon.pfIsoId >= cut_mapping[self.iso_working_point]
            passed &= pass_id_wp

        columns[self.output_col] = muon[passed]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
