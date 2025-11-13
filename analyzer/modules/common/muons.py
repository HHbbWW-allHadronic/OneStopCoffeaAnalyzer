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


class IsoWps(enum.Enum):
    very_loose = 1
    loose = 2
    medium = 3
    tight = 4
    very_tight = 5
    very_very_tight = 5


@define
class MuonMaker(AnalyzerModule):
    input_col: Column
    output_col: Column
    id_working_point: IdWps
    iso_working_point: IsoWps
    min_pt: float = 10
    max_abs_eta: float = 2.4

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        muon = columns[self.input_col]
        pass_pt = muon.pt > self.min_pt
        pass_eta = abs(muon.eta) < self.max_abs_eta
        pass_id_wp = muon[self.id_working_point]
        pass_iso_wp = muon.pfIsoId >= self.iso_working_point
        columns[self.output_col] = muon[pass_pt & pass_eta & pass_id_wp & pass_iso_wp]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
