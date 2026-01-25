from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr
import pickle
import warnings
from analyzer.core.columns import Column
import awkward as ak
from attrs import define
from ..common.axis import RegularAxis
from ..common.histogram_builder import makeHistogram
import numpy as np


@define
class NNMassReco(AnalyzerModule):
    """
    Reconstruct top quark and neutralino masses using a Neural Network.

    This module uses a trained PyTorch model to resolve jet combinatorics and
    reconstruct the mass of the top quark ($m_{\\tilde{t}}$) and neutralino ($m_{\\chi}$).

    Parameters
    ----------
    input_col : Column
        Input column containing the jet collection (e.g. GoodJet).
    m3_output : Column
        Output column name for the reconstructed neutralino mass ($m_{\\chi}$).
    m4_output : Column
        Output column name for the reconstructed top quark mass ($m_{\\tilde{t}}$).
    model_path : str
        Path to the trained PyTorch model file (.pt).
    scaler_path : str
        Path to the scaler file (.pkl) used for input feature normalization.

    """

    input_col: Column
    m3_output: Column
    m4_output: Column

    model_path: str
    scaler_path: str

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.m3_output, self.m4_output]

    def neededResources(self, metadata):
        return [self.model_path, self.scaler_path]

    def run(self, columns, params):
        from analyzer.coffea_patches.torch_wrapper import torch_wrapper

        class jetAssignmentNN(torch_wrapper):
            def prepare_awkward(_, jets, scalerFile, _fake):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(scalerFile, "rb") as f:
                        scaler = pickle.load(f)

                # jets = columns[self.input_col]
                flat_jets = ak.flatten(jets)
                m3 = jets[:, 1:4].sum()
                m4 = jets[:, 0:4].sum()

                ones = ak.ones_like(jets.pt)

                if "binaryNominal_24-09-11-17-46" not in scalerFile:
                    imap = {
                        "features": {
                            "jetOrdinality": ak.flatten(ak.local_index(jets, axis=1)),
                            "jetPT": flat_jets.pt,
                            "jetEta": flat_jets.eta,
                            "jetPhi": flat_jets.phi,
                            "jetBScore": flat_jets.btagDeepFlavB,
                            "m3M": ak.flatten(ones * m3.mass),
                            "m3PT": ak.flatten(ones * m3.pt),
                            "m3Eta": ak.flatten(ones * m3.eta),
                            "m3Phi": ak.flatten(ones * m3.phi),
                            "m4M": ak.flatten(ones * m4.mass),
                            "m4PT": ak.flatten(ones * m4.pt),
                            "m4Eta": ak.flatten(ones * m4.eta),
                            "m4Phi": ak.flatten(ones * m4.phi),
                            "nJets": ak.flatten(ones * ak.num(jets)),
                        }
                    }
                else:
                    imap = {
                        "features": {
                            "jetOrdinality": ak.flatten(ak.local_index(jets, axis=1)),
                            "jetPT": flat_jets.pt,
                            "jetEta": flat_jets.eta,
                            "jetPhi": flat_jets.phi,
                            "jetBScore": flat_jets.btagDeepFlavB,
                            "m3M": ak.flatten(ones * m3.mass),
                            "m3PT": ak.flatten(ones * m3.pt),
                            "m3Eta": ak.flatten(ones * m3.eta),
                            "m3Phi": ak.flatten(ones * m3.phi),
                            "m4M": ak.flatten(ones * m4.mass),
                            "m4PT": ak.flatten(ones * m4.pt),
                            "m4Eta": ak.flatten(ones * m4.eta),
                            "m4Phi": ak.flatten(ones * m4.phi),
                        }
                    }

                imap_concat = ak.concatenate(
                    [x[:, np.newaxis] for x in imap["features"].values()], axis=1
                )
                imap_scaled = (imap_concat - scaler.mean_) / scaler.scale_
                return (), {"x": ak.values_astype(imap_scaled, "float32")}

        jets = columns[self.input_col]
        model = jetAssignmentNN(self.model_path)
        outputs = model(jets, self.scaler_path, columns._events)[:, 0]
        # m14 = jets[:, 0:4].sum().mass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            top_3_idx = ak.argsort(ak.unflatten(outputs, ak.num(jets)), axis=1)[:, -3:]
        mChiComp = jets[top_3_idx].sum()
        top_3_excl_mask = (
            (ak.local_index(jets, axis=1) != top_3_idx[:, 0])
            & (ak.local_index(jets, axis=1) != top_3_idx[:, 1])
            & (ak.local_index(jets, axis=1) != top_3_idx[:, 2])
        )
        stop_b = jets[top_3_excl_mask][:, 0]  # Highest remaining pT
        m14 = stop_b + mChiComp

        chi_m = mChiComp.mass
        stop_m = m14.mass
        columns[self.m3_output] = chi_m
        columns[self.m4_output] = stop_m
        return columns, []


@define
class NNMassPlots(AnalyzerModule):
    """
    Create histograms for reconstructed mass variables.

    Generates 1D and 2D histograms for the reconstructed top squark and chargino masses,
    as well as their ratio.

    Parameters
    ----------
    m3_input : Column
        Column containing the reconstructed chargino mass ($m_{\\chi}$).
    m4_input : Column
        Column containing the reconstructed top squark mass ($m_{\\tilde{t}}$).
    prefix : str
        Prefix required for all generated histograms to ensure uniqueness.
    """

    m3_input: Column
    m4_input: Column
    prefix: str

    def run(self, columns, params):
        chi_m = columns[self.m3_input]
        stop_m = columns[self.m4_input]

        ret = []
        ret.append(
            makeHistogram(
                f"{self.prefix}_mChi",
                columns,
                RegularAxis(60, 0, 3000, r"$m_{\tilde{t}}$", unit="GeV"),
                chi_m,
            )
        )
        ret.append(
            makeHistogram(
                f"{self.prefix}_mStop",
                columns,
                RegularAxis(60, 0, 3000, r"$m_{\tilde{t}}$", unit="GeV"),
                stop_m,
            )
        )
        ret.append(
            makeHistogram(
                f"{self.prefix}_mStop_vs_mChi",
                columns,
                [
                    RegularAxis(60, 0, 3000, r"$m_{\tilde{t}}$", unit="GeV"),
                    RegularAxis(60, 0, 3000, r"$m_{\chi}$", unit="GeV"),
                ],
                [stop_m, chi_m],
            )
        )
        ret.append(
            makeHistogram(
                f"{self.prefix}_mStop_vs_mChiRatio",
                columns,
                [
                    RegularAxis(60, 0, 3000, r"$m_{\tilde{t}}$", unit="GeV"),
                    RegularAxis(50, 0, 1, r"$m_{\chi} / m_{\tilde{t}}$", unit="GeV"),
                ],
                [stop_m, chi_m / stop_m],
            )
        )
        return columns, ret

    def outputs(self, metadata):
        return []

    def inputs(self, metadata):
        return [self.m3_input, self.m4_input]
