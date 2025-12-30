import logging
import itertools as it
import pickle as pkl
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
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    register_module,
    MetadataExpr,
    ParameterSpec,
    ModuleParameterSpec,
    IsSampleType,
)
from analyzer.core.columns import Column
import awkward as ak
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram


@define
class BJetShapeSF(AnalyzerModule):
    input_col: Column
    weight_name: str = "b_tag_disc_shape"

    should_run: MetadataExpr = field(factory=lambda: IsSampleType("MC"))

    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        b_meta = metadata["era"]["btag_scale_factors"]
        systematics = b_meta["systematics"]
        possible_values = it.product(["up", "down"], systematics)
        possible_values = ["central"] + [
            f"{updown}_{name}" for updown, name in possible_values
        ]

        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=possible_values,
                    tags={"weight_variation"},
                ),
            }
        )

    def run(self, columns, params):
        sf_eval = self.getCorrection(columns.metadata)
        systematic = params["variation"]
        gj = columns[self.input_col]
        # bjets = jets[jets.btagDeepFlavB > wp[self.working_point]]
        if systematic == "central":
            j = gj
            sf = ak.prod(
                sf_eval.evaluate(
                    "central", j.hadronFlavour, abs(j.eta), j.pt, j.btagDeepFlavB
                ),
                axis=1,
            )
        elif "_cf" in systematic:
            j = gj[gj.hadronFlavour == 4]
            sf = ak.prod(
                sf_eval.evaluate(
                    systematic, j.hadronFlavour, abs(j.eta), j.pt, j.btagDeepFlavB
                ),
                axis=1,
            )
        else:
            j = gj[gj.hadronFlavour != 4]
            sf = ak.prod(
                sf_eval.evaluate(
                    systematic, j.hadronFlavour, abs(j.eta), j.pt, j.btagDeepFlavB
                ),
                axis=1,
            )

        columns["Weights", self.weight_name] = sf
        return columns, []

    def getCorrection(self, metadata):
        file_path = metadata["era"]["btag_scale_factors"]["file"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = cset["deepJet_shape"]
        self.__corrections[file_path] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Weights", self.weight_name))]
