import awkward as ak
from analyzer.core import AnalyzerModule


@define
class JetFilter(AnalyzerModule):
    input_col: Column
    output_col: Column
    min_pt: float = 30.0
    max_abs_eta: float = 2.4
    include_pu_id: bool = False
    include_jet_id: bool = False

    def run(self, columns, params):
        metadata = columns.metadata
        jets = columns[self.input_col]
        good_jets = jets[(jets.pt > self.min_pt) & (abs(jets.eta) < self.max_abs_eta)]

        if self.include_jetid:
            good_jets = good_jets[
                ((good_jets.jetId & 0b100) != 0) & ((good_jets.jetId & 0b010) != 0)
            ]

        if self.include_puid:
            if any(x in metadata.dataset.era.name for x in ["2016", "2017", "2018"]):
                good_jets = good_jets[(gj.pt > 50) | ((good_jets.puId & 0b10) != 0)]
        columns[self.output_col] = good_jets
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class JetFilter(AnalyzerModule):
    input_col: Column
    output_col: Column
    min_pt: float = 30.0
    max_abs_eta: float = 2.4

    def run(self, columns, params):
        jets = columns[self.input_col]
        good_jets = jets[(jets.pt > self.min_pt) & (abs(jets.eta) < self.max_abs_eta)]

        columns[self.output_col] = good_jets
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
