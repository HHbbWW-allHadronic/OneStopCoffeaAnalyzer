import coffea.lumi_tools as ltools
from analyzer.configuration import CONFIG
from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
import enum
import correctionlib
from pathlib import Path
import fsspec
import awkward as ak
import correctionlib
import correctionlib.convert
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from correctionlib.convert import from_histogram
import re
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
import correctionlib


@define
class PileupSF(AnalyzerModule):
    input_col: Column
    weight_name: str = "pileup_sf"

    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up", "down"],
                    tags={
                        "weight_variation",
                    },
                ),
            }
        )

    def run(self, columns, params):
        corr = self.getCorrection(columns.metadata)
        n_pu = columns["Pileup"]["nTrueInt"]
        correction = corr.evaluate(n_pu, params["variation"])
        columns[Column(("Weight", self.weight_name))] = correction
        return columns, []

    def getCorrection(self, metadata):
        file_path = metadata["era"]["pileup_scale_factors"]["file"]
        name = metadata["era"]["pileup_scale_factors"]["name"]
        if (name, file_path) in self.__corrections:
            return self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = cset[name]
        self.__corrections[(name, file_path)] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [Column("Pileup.nTrueInt")]

    def outputs(self, metadata):
        return [Columns(fields=("Weights", self.weight_name))]


@define
class L1PrefiringSF(AnalyzerModule):
    input_col: Column
    weight_name: str = "l1_prefiring"

    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="Nom",
                    possible_values=["Nom", "Up", "Down"],
                    tags={"weight_variation"},
                ),
            }
        )

    def run(self, columns, params):
        variation = params["variation"]
        columns["Weights", self.weight_name] = columns["L1PreFiringWeight"][variation]
        return columns, []

    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return [Columns(fields=("Weights", self.weight_name))]


@define
class GoldenLumi(AnalyzerModule):
    selection_name: str = "golden_lumi"

    def inputs(self, metadata):
        return [Column("run"), Column("luminosityBlock")]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]

    def run(events, params, selector):
        lumi_json = params.dataset.era.golden_json
        lmask = ltools.LumiMask(params["golden_json"])
        selector.add(f"golden_lumi", lmask(events["run"], events["luminosityBlock"]))
