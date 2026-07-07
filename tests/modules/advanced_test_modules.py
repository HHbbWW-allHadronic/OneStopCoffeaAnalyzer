from analyzer.core.analysis_modules import (
    AnalyzerModule,
    ModuleParameterSpec,
    ParameterSpec,
)
from analyzer.core.columns import Column
from attrs import define, field
import awkward as ak
import numpy as np
from typing import ClassVar


@define
class ObjectProducer(AnalyzerModule):
    input_col: Column
    output_col: Column

    def run(self, columns, params):
        objs = columns[self.input_col]
        # Just pass through or do something simple
        columns[self.output_col] = objs
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class DependentModule(AnalyzerModule):
    input_col: Column
    output_col: Column

    def run(self, columns, params):
        objs = columns[self.input_col]
        # Depends on output of ObjectProducer
        columns[self.output_col] = objs.pt * 2
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class CorrelatedSystA(AnalyzerModule):
    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "param_A": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up", "down"],
                    tags={"shape_variation"},
                )
            }
        )

    def run(self, columns, params):
        # We'll just add a column to track which variation ran
        variation = params["param_A"]
        columns["param_A_val"] = variation
        return columns, []

    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return [Column("param_A_val")]


@define
class CorrelatedSystB(AnalyzerModule):
    def getParameterSpec(self, metadata):
        def mapAtoB(val):
            if val == "nominal":
                return None
            return val  # Same value for simplicity

        return ModuleParameterSpec(
            {
                "param_B": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up", "down"],
                    tags={"weight_variation"},
                    driven_by={"param_A": mapAtoB},
                )
            }
        )

    def run(self, columns, params):
        variation = params["param_B"]
        columns["param_B_val"] = variation
        return columns, []

    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return [Column("param_B_val")]


@define
class ExecutionCounter(AnalyzerModule):
    counts: ClassVar[dict] = {}

    counter_name: str

    def run(self, columns, params):
        if self.counter_name not in ExecutionCounter.counts:
            ExecutionCounter.counts[self.counter_name] = 0
        ExecutionCounter.counts[self.counter_name] += 1
        return columns, []

    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return []

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec({})


@define
class IndependentWeightSyst(AnalyzerModule):
    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "independent_weight": ParameterSpec(
                    default_value="nominal",
                    possible_values=["nominal", "up", "down"],
                    tags={"weight_variation"},
                )
            }
        )

    def run(self, columns, params):
        variation = params["independent_weight"]
        if variation == "up":
            weight = 1.1
        elif variation == "down":
            weight = 0.9
        else:
            weight = 1.0
        # Use Muon as representative to get the correct length
        n_events = len(columns["Muon"])
        weight_array = ak.ones_like(range(n_events), dtype=float) * weight
        columns["Weights", "independent_weight"] = weight_array
        return columns, []

    def inputs(self, metadata):
        return [Column("Muon")]

    def outputs(self, metadata):
        return [Column("Weights.independent_weight")]
