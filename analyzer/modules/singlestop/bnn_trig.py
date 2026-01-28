import json
from pathlib import Path
import correctionlib

import awkward as ak
import numpy as np
from attrs import define, field

from analyzer.core.analysis_modules import (
    AnalyzerModule,
    IsSampleType,
    MetadataExpr,
    ModuleParameterSpec,
    ParameterSpec,
)
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import SimpleCache


class BNNEnsemble:
    def __init__(self, num_inputs, mean, sigma, hidden_width, data):
        self.mean = mean
        self.sigma = sigma
        hidden_length = hidden_width * (num_inputs + 1)
        hidden_data = data[:, 0:hidden_length].reshape(len(data), -1, num_inputs + 1)
        self.output_bias = data[:, hidden_length]
        self.output_weights = data[:, hidden_length + 1 :]
        self.hidden_biases = hidden_data[:, :, 0]
        self.hidden_weights = hidden_data[:, :, 1:]

    def __call__(self, data):
        data = data.to_numpy()
        data = (data - self.mean) / self.sigma
        l1 = self.hidden_biases + np.matvec(self.hidden_weights, data[:, None])
        l2 = self.output_bias + np.vecdot(self.output_weights, np.tanh(l1))
        vals = 1 / (1 + np.exp(-l2))
        return (
            np.median(vals, axis=1),
            np.percentile(vals, 15.865, axis=1),
            np.percentile(vals, 84.135, axis=1),
        )

    @staticmethod
    def fromFile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return BNNEnsemble(
            data["num_inputs"],
            np.array(data["mean"]),
            np.array(data["sigma"]),
            data["hidden_width"],
            np.array(data["parameters"]),
        )


@define
class TriggerBNN(AnalyzerModule):
    """
    Compute trigger efficiency weights using a BNN ensemble.

    This analyzer evaluates a Bayesian Neural Network on HT and leading
    fat jet pt to produce a trigger weight for MC samples.

    Parameters
    ----------
    base_path : str
        Base directory where BNN JSON files are stored.
    net_pattern : str
        Pattern for the network filename, formatted with the era name.
    weight_name : str, optional
        Name of the output weight column, by default "trigger_eff".
    should_run : MetadataExpr, optional
        Condition to determine if the module should run. By default runs on MC samples.
    """

    base_path: str
    net_pattern: str
    weight_name: str = "trigger_eff"

    should_run: MetadataExpr = field(factory=lambda: IsSampleType("MC"))

    __bnns: dict = field(factory=dict)
    __bnn_res_cache: dict = field(factory=SimpleCache)

    def inputs(self, metadata):
        return [Column("HT"), Column("GoodFatJet")]

    def outputs(self, metadata):
        return [Column(fields=("Weights", self.weight_name))]

    def neededResources(self, metadata):
        return [self.base_path]

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=["central", "up", "down"],
                    tags={
                        "weight_variation",
                    },
                ),
            }
        )

    def getBNN(self, metadata):
        name = metadata["era"]["name"]
        path = str(Path(self.base_path) / self.net_pattern.format(era=name))
        if path in self.__bnns:
            return self.__bnns[path]
        bnn = BNNEnsemble.fromFile(path)
        self.__bnns[path] = bnn
        return bnn

    def run(self, columns, params):
        k = self.getKeyNoParams(columns)
        systematic = params["variation"]
        if k in self.__bnn_res_cache:
            central, down, up = self.__bnn_res_cache[k]
        else:
            ht = columns["HT"]
            fj = columns["GoodFatJet"]
            fjpt = fj[:, 0].pt
            bnn = self.getBNN(columns.metadata)
            out = bnn(ak.concatenate([x[:, np.newaxis] for x in [ht, fjpt]], axis=1))
            central, down, up = out[0], out[1], out[2]

        self.__bnn_res_cache[k] = (central, down, up)
        if systematic == "central":
            w = central
        elif systematic == "up":
            w = up
        elif systematic == "down":
            w = down
        columns[Column(("Weights", self.weight_name))] = w
        return columns, []


@define
class TriggerBNNCorrection(AnalyzerModule):
    """
    Compute trigger efficiency weights using correctionlib for BNN.

    Parameters
    ----------
    base_path : str
        Base directory where correction files are stored.
    correction_pattern : str
        Pattern for the correction filename, formatted with the era name (e.g. bnn_correction_{era}.json.gz).
    correction_name : str, optional
        Name of the correction in the file, by default "BNN_Trigger_Efficiency".
    weight_name : str, optional
        Name of the output weight column, by default "trigger_eff".
    should_run : MetadataExpr, optional
        Condition to determine if the module should run. By default runs on MC samples.
    """

    base_path: str
    correction_pattern: str
    correction_name: str = "BNN_Trigger_Efficiency"
    weight_name: str = "trigger_eff"

    should_run: MetadataExpr = field(factory=lambda: IsSampleType("MC"))

    __corrections: dict = field(factory=dict)

    def inputs(self, metadata):
        return [Column("HT"), Column("GoodFatJet")]

    def outputs(self, metadata):
        return [Column(fields=("Weights", self.weight_name))]

    def neededResources(self, metadata):
        return [self.base_path]

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="central",
                    possible_values=["central", "up", "down"],
                    tags={
                        "weight_variation",
                    },
                ),
            }
        )

    def getCorrection(self, metadata):
        name = metadata["era"]["name"]
        path = str(Path(self.base_path) / self.correction_pattern.format(era=name))

        if path in self.__corrections:
            return self.__corrections[path]

        cset = correctionlib.CorrectionSet.from_file(path)
        corr = cset[self.correction_name]
        self.__corrections[path] = corr
        return corr

    def run(self, columns, params):
        systematic = params["variation"]

        syst_map = {"central": "nominal", "up": "up", "down": "down"}

        corr_syst = syst_map.get(systematic, "nominal")

        ht = columns["HT"]
        fj = columns["GoodFatJet"]
        fjpt = fj[:, 0].pt

        corr = self.getCorrection(columns.metadata)

        w = corr.evaluate(corr_syst, ht, fjpt)

        columns[Column(("Weights", self.weight_name))] = w
        return columns, []
