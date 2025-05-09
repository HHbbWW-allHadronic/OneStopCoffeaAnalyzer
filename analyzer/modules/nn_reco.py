import pickle

import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType


@MODULE_REPO.register(ModuleType.Histogram)
def NN_mass_reco(
    events, params, analyzer, model_path=None, scaler_path=None, model_name=""
):

    from coffea.ml_tools.torch_wrapper import torch_wrapper
    import numpy as np

    from .utils.axes import makeAxis

    class jetAssignmentNN(torch_wrapper):
        def prepare_awkward(self, events, scalerFile, _fake):

            with open(scalerFile, "rb") as f:
                scaler = pickle.load(f)

            jets = events.good_jets
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

    if model_path is None or scaler_path is None:
        raise ValueError("NN Mass Reco Requires a model path and scaler path")
    jets = events.good_jets
    model = jetAssignmentNN(model_path)
    outputs = model(events, scaler_path, events.events)[:, 0]
    # m14 = jets[:, 0:4].sum().mass
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

    analyzer.H(
        f"{model_name}_mChi",
        makeAxis(
            150,
            0,
            3000,
            rf"mChi",
            unit="GeV",
        ),
        chi_m,
    )
    analyzer.H(
        f"{model_name}_mStop",
        makeAxis(
            150,
            0,
            3000,
            rf"mStop",
            unit="GeV",
        ),
        stop_m,
    )
    analyzer.H(
        f"{model_name}_m14_vs_mChi",
        [
            makeAxis(125, 500, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(125, 0.15, 3000, f"$m_{{3 (NN, {model_name})}}$", unit="GeV"),
        ],
        [stop_m, chi_m],
    )
    analyzer.H(
        f"{model_name}_m14_vs_mChiRatio",
        [
            makeAxis(125, 500, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(
                125, 0.15, 1, f"$m_{{3 (NN, {model_name})}} / m_{{14}}$", unit="GeV"
            ),
        ],
        [stop_m, chi_m / stop_m],
    )
