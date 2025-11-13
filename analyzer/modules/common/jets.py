from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
import awkward as ak
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram


@define
class VetoMapFilter(AnalyzerModule):
    input_col: Column
    output_col: Column
    # should_run: MetadataFunc = field(factory=lambda: IS_RUN_2)

    def run(self, columns, params):
        map_name = columns.metadata["era"]["jet_veto_map"]["name"]
        corr = self.getCorr(columns.metadata)[map_name]
        j = columns[self.input_col]
        j = j[
            (abs(j.eta) < 2.4)
            & (j.pt > 15)
            & ((j.jetId & 0b100) != 0)
            & ((j.chEmEF + j.neEmEF) < 0.9)
        ]
        vetoes = eval_veto.evaluate(veto_type, j.eta, j.phi)
        passed_jets = j[(vetoes == 0)]
        columns[self.output_col] = passed_jets
        return columns, []

    def getCorr(self, metadata):
        file_path = metadata["era"]["jet_veto_map"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)
        self.__corrections[file_path] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorr(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

@define
class HT(AnalyzerModule):
    input_col: Column
    output_col: Column = field(factory=lambda: Column("HT"))

    def run(self, columns, params):
        jets = columns[self.input_col]
        columns[self.output_col] = ak.sum(jets.pt, axis=1)
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
            if any(
                x in metadata["dataset"]["era"]["name"]
                for x in ["2016", "2017", "2018"]
            ):
                good_jets = good_jets[(gj.pt > 50) | ((good_jets.puId & 0b10) != 0)]
        columns[self.output_col] = good_jets
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@register_module(
    configuration={
        "prefix": field(type=str),
        "input_col": field(type=Column),
        "max_idx": field(type=int),
        "include_properties": field(
            type=list[str], factory=lambda: ["pt", "eta", "phi"]
        ),
    },
    input_columns=lambda self: [self.input_col],
    output_columns=lambda self: [],
)
def TopJetHistograms(self, columns, params):
    jets = columns[self.input_col]
    ret = []
    for i in range(0, self.max_idx):
        mask = ak.num(jets, axis=1) > i
        masked_jets = jets[mask]
        if "pt" in self.include_properties:
            ret.append(
                makeHistogram(
                    f"{self.prefix}_pt_{i+1}",
                    columns,
                    RegularAxis(20, 0, 1000, f"$p_{{T, {i+1}}}$", unit="GeV"),
                    masked_jets[:, i].pt,
                    description=f"$p_T$ of jet {i+1} ",
                    mask=mask,
                )
            )
    return columns, ret
