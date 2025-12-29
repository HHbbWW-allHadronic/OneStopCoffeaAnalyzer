import coffea.lumi_tools as ltools
import functools as ft
from analyzer.core.columns import addSelection
import operator as op
from analyzer.configuration import CONFIG
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    register_module,
    ModuleParameterSpec,
    ParameterSpec,
    MetadataExpr,
    MetadataAnd,
    IsRun,
    IsSampleType,
)
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
    weight_name: str = "pileup_sf"

    should_run: MetadataExpr = field(factory=lambda: IsSampleType("MC"))
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
        n_pu = columns["Pileup.nTrueInt"]
        correction = corr.evaluate(n_pu, params["variation"])
        columns[Column(("Weights", self.weight_name))] = correction
        return columns, []

    def getCorrection(self, metadata):
        file_path = metadata["era"]["pileup_scale_factors"]["file"]
        name = metadata["era"]["pileup_scale_factors"]["name"]
        key = (name, file_path)
        if key in self.__corrections:
            return self.__corrections[key]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = cset[name]
        self.__corrections[key] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [Column("Pileup.nTrueInt")]

    def outputs(self, metadata):
        return [Column(fields=("Weights", self.weight_name))]


@define
class L1PrefiringSF(AnalyzerModule):
    should_run: MetadataExpr = field(
        factory=lambda: MetadataAnd([IsSampleType("MC"), IsRun(2)])
    )
    weight_name: str = "l1_prefiring"

    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="Nom",
                    possible_values=["Nom", "Up", "Dn"],
                    tags={"weight_variation"},
                ),
            }
        )

    def run(self, columns, params):
        variation = params["variation"]
        columns["Weights", self.weight_name] = columns["L1PreFiringWeight"][variation]
        return columns, []

    def inputs(self, metadata):
        return [Column("L1PreFiringWeight")]

    def outputs(self, metadata):
        return [Column(fields=("Weights", self.weight_name))]


@define
class GoldenLumi(AnalyzerModule):
    selection_name: str = "golden_lumi"
    should_run: MetadataExpr = field(factory=lambda: IsSampleType("Data"))

    def inputs(self, metadata):
        return [Column("run"), Column("luminosityBlock")]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]

    def run(self, columns, params):
        metadata = columns.metadata
        lumi_json = metadata["era"]["golden_json"]
        lmask = ltools.LumiMask(lumi_json)
        addSelection(
            columns,
            self.selection_name,
            lmask(columns["run"], columns["luminosityBlock"]),
        )
        return columns, []


@define
class NoiseFilter(AnalyzerModule):
    selection_name: str = "noise_filters"

    def inputs(self, metadata):
        return [Column("Flag")]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]

    def run(self, columns, params):
        metadata = columns.metadata
        noise_flags = metadata["era"]["noise_filters"]
        sel = ft.reduce(op.and_, [columns["Flag"][x] for x in noise_flags])
        addSelection(columns, self.selection_name, sel)
        return columns, []
