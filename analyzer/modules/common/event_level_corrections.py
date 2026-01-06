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
    """
    Apply pileup scale factors to Monte Carlo events.

    This analyzer evaluates the pileup weight based on the number of true
    interactions in the event and applies systematic variations if requested.

    Parameters
    ----------
    weight_name : str, optional
        Name of the column where the pileup weights are stored, by default "pileup_sf".
    should_run : MetadataExpr, optional
        Condition to determine if the module should run. By default runs only
        on MC samples.
    """

    weight_name: str = "pileup_sf"

    should_run: MetadataExpr = field(factory=lambda: IsSampleType("MC"))
    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up", "down", "disabled"],
                    tags={
                        "weight_variation",
                    },
                ),
            },
        )

    def run(self, columns, params):
        if params["variation"] == "disabled":
            return columns, []
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
    """
    Apply L1 prefiring scale factors to Monte Carlo events.

    This analyzer retrieves the L1 prefiring weight and adds it to a named weight column, optionally applying systematic variations.

    Parameters
    ----------
    weight_name : str, optional
        Name of the output weight column, by default "l1_prefiring".
    should_run : MetadataExpr, optional
        Condition to determine if the module should run. By default, it runs
        on MC samples for Run 2.
    """
    weight_name: str = "l1_prefiring"
    should_run: MetadataExpr = field(
        factory=lambda: MetadataAnd([IsSampleType("MC"), IsRun(2)])
    )

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
class PosNegGenWeight(AnalyzerModule):
    should_run: MetadataExpr = field(factory=lambda: IsSampleType("MC"))
    weight_name: str = "pos_neg_weight"

    def run(self, columns, params):
        gw = ak.where(columns["genWeight"] > 0, 1.0, -1.0)
        columns["Weights", self.weight_name] = gw
        return columns, []

    def inputs(self, metadata):
        return [Column("genWeight")]

    def outputs(self, metadata):
        return [Column(fields=("Weights", self.weight_name))]


@define
class GoldenLumi(AnalyzerModule):
    """
    Apply a golden JSON luminosity selection for data events.

    This analyzer filters events to only include those that are within
    certified good luminosity sections for the given data-taking era.

    Parameters
    ----------
    selection_name : str, optional
        Name of the selection column to store the golden JSON filter,
        by default "golden_lumi".
    should_run : MetadataExpr, optional
        Condition to determine if the module should run. By default, only
        runs on real data samples.
    Notes
    -----
    - The certified luminosity sections are read from the metadata for
        the given era under "golden_json".
    """
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
    """
    Apply standard noise filters to data events.

    This analyzer combines multiple event-level noise flags and produces a
    single selection column indicating events that pass all required noise
    filters.

    Parameters
    ----------
    selection_name : str, optional
        Name of the selection column to store the combined noise filter,
        by default "noise_filters".
    """

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
