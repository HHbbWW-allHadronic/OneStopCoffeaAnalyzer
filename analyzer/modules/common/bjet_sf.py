import logging
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
from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
import awkward as ak
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram


@define
class BJetShapeSF(AnalyzerModule):
    input_col: Column
    weight_name: str = "b_tag_disc_shape"

    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=["central", "up", "down"],
                    tags={
                        "weight_variation",
                    },
                    metadata={"correlation_function": None},
                ),
            }
        )

    def run(self, columns, params):
        wps = self.getWPs(columns.metadata)
        jets = columns[self.input_col]
        bjets = jets[jets.btagDeepFlavB > wp[self.working_point]]
        columns[self.output_col] = bjets
        return columns, []

    def getCorrection(self, metadata):
        file_path = metadata["era"]["btag_scale_factors"]["file"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        era_info = params.dataset.era
        cset = getBTagCset(era_info.btag_scale_factors["file"])
        self.__corrections[file_path] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
