from analyzer.core.analysis_modules import (
    AnalyzerModule,
    ModuleParameterSpec,
    ParameterSpec,
)
from analyzer.core.columns import Column
from attrs import define
import awkward as ak
import numpy as np


@define
class MuonScaleSyst(AnalyzerModule):
    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up", "down"],
                    tags={"shape_variation"},
                ),
            }
        )

    def inputs(self, metadata):
        return [Column("Muon.pt")]

    def outputs(self, metadata):
        return [Column("Muon.pt")]

    def run(self, columns, params):
        muon_pt = columns["Muon.pt"]
        variation = params["variation"]

        factor = 1.0
        if variation == "up":
            factor = 1.05
        elif variation == "down":
            factor = 0.95

        if factor != 1.0:
            scaled_pt = muon_pt * factor
            columns["Muon.pt"] = scaled_pt
        return columns, []


@define
class MuonResSyst(AnalyzerModule):
    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up"],
                    tags={"shape_variation"},
                ),
            }
        )

    def inputs(self, metadata):
        return [Column("Muon.pt")]

    def outputs(self, metadata):
        return [Column("Muon.pt")]

    def run(self, columns, params):
        muon_pt = columns["Muon.pt"]
        variation = params["variation"]

        sigma = 0.0
        if variation == "up":
            sigma = 0.1

        if sigma > 0:
            rng = np.random.default_rng(42)
            counts = ak.num(muon_pt)
            flat_pt = ak.flatten(muon_pt)
            smear = rng.normal(0, sigma, size=len(flat_pt))
            flat_scaled = flat_pt * (1 + smear)
            smeared_pt = ak.unflatten(flat_scaled, counts)
            columns["Muon.pt"] = smeared_pt

        return columns, []


@define
class EventWeightSyst(AnalyzerModule):
    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up"],
                    tags={"weight_variation"},
                ),
            }
        )

    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return [Column("Weights")]

    def run(self, columns, params):
        variation = params["variation"]

        factor = 1.0
        if variation == "up":
            factor = 1.5

        n_events = len(columns["Muon"])
        w_array = ak.ones_like(range(n_events), dtype=float) * factor

        # Always add/create weight column, even if 1.0
        # This ensures 'Weights' group exists and provenance is updated
        columns["Weights", "syst_weight"] = w_array
        return columns, []


@define
class TestCachingModule(AnalyzerModule):
    execution_counts: list = []

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec({})

    def inputs(self, metadata):
        return [Column("Muon.pt")]

    def outputs(self, metadata):
        return [Column("dummy_col")]

    def run(self, columns, params):
        # Track execution context (e.g. mean pt) to verify distinct inputs
        pt_mean = float(ak.mean(ak.flatten(columns["Muon.pt"])))

        self.execution_counts.append({"params": params, "pt_mean": pt_mean})

        columns["dummy_col"] = ak.zeros_like(columns["Muon.pt"])  # Safe dummy
        return columns, []
