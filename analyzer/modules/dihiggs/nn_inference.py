from coffea.ml_tools.torch_wrapper import torch_wrapper
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define
import awkward as ak
import numpy as np
import torch


@define
class ABCDiHiggsInference(AnalyzerModule):
    r"""
    Inference module for the ABCDiHiggs neural network.
    This module takes in jet-like columns and global event features,
    prepares the inputs for the neural network, performs inference,
    and adds the output discriminators to the columns.

    Parameters
    ----------
    jet_col : Column
        Column containing the jet collection.
    jet_vars : list[str]
        List of jet variables to be used as inputs for the neural network.
    global_cols : list[Column]
        List of global event feature columns to be used as inputs for the neural network.
    model_path : str
        Path to the trained neural network model file.
    scaler_path : str
        Path to the scaler file used for input normalization.
    output_cols : list[Column]
        List of columns to store the output discriminators from the neural network.
    n_jets : int, optional
        Number of jets to prepare for the input, default is 6.
    """

    jet_col: Column
    jet_vars: list[str]
    global_cols: list[Column]
    model_path: str
    scaler_path: str
    output_cols: list[Column]
    n_jets: int = 6

    def prepare_inputs(self, columns):
        inputs = []
        for var in self.jet_vars:
            field = Column(
                var
                if (var != "btag")
                else columns.metadata["era"]["btag_scale_factors"]["tagger"]
            )
            in_var = columns[self.jet_col + field]
            padded = ak.pad_none(in_var, self.n_jets, clip=True)
            padded = ak.fill_none(padded, 0)
            inputs.append(padded)
        for global_col in self.global_cols:
            inputs.append(columns[global_col][:, np.newaxis])
        return ak.concatenate(inputs, axis=1)

    def run(self, columns, params):
        n_discs = len(self.output_cols)

        class ABCDiHiggsNetwork(torch_wrapper):
            def prepare_awkward(self, inputs):
                return [
                    ak.values_astype(inputs, "float32"),
                ], {}

            def postprocess_awkward(self, output, events):
                ret = {f"Disc{i}": output[:, i] for i in range(n_discs)}
                return ret

        model = ABCDiHiggsNetwork(self.model_path)
        X = self.prepare_inputs(columns)
        with open(self.scaler_path, "rb") as f:
            scaler = torch.load(f, map_location="cpu", weights_only=False)["scaler"]
        X = (X - scaler.mean_) / scaler.scale_

        if len(X) == 0:
            empty = np.array([], dtype="float32")
            for col in self.output_cols:
                columns[col + Column("sig")] = ak.Array(empty)
                columns[col + Column("qcd")] = ak.Array(empty)
                columns[col + Column("tt")] = ak.Array(empty)
            return columns, []

        outputs = model(X)
        for i, col in enumerate(self.output_cols):
            columns[col + Column("sig")] = outputs[f"Disc{i}"][:, 0]
            columns[col + Column("qcd")] = outputs[f"Disc{i}"][:, 1]
            columns[col + Column("tt")] = outputs[f"Disc{i}"][:, 2]
        return columns, []

    def outputs(self, metadata):
        return self.output_cols

    def inputs(self, metadata):
        return [self.jet_col] + self.global_cols
