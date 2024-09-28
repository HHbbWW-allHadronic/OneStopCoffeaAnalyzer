import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType
from .utils.axes import makeAxis


    

b_tag_wps = [0.0490, 0.2783, 0.7100]

@MODULE_REPO.register(ModuleType.Histogram)
def NN_mass_reco(events, params, analyzer, model_path=None, model_name=""):
    from coffea.ml_tools.torch_wrapper import torch_wrapper
    class jetAssignmentNN(torch_wrapper):
        def prepare_awkward(self,events):
            # ak = self.get_awkward_lib(events)
            jets = events.good_jets
            flat_jets = ak.flatten(jets)

            m3 = jets[:,1:4].sum()
            m4 = jets[:,0:4].sum()

            ones = ak.ones_like(jets.pt)

            imap = {
                "features": {
                    "jetOrdinality":    ak.flatten(ak.local_index(jets, axis=1)),
                    "jetPT": 		    flat_jets.pt - 2,
                    "jetEta": 		    flat_jets.eta,
                    "jetPhi": 		    flat_jets.phi,
                    "jetBScore":    	flat_jets.btagDeepFlavB,
                    "m3M": 			    ak.flatten(ones * m3.mass),
                    "m3PT": 		    ak.flatten(ones * m3.pt),
                    "m3Eta": 		    ak.flatten(ones * m3.eta),
                    "m3Phi": 	        ak.flatten(ones * m3.phi),
                    "m4M": 			    ak.flatten(ones * m4.mass),
                    "m4PT":		        ak.flatten(ones * m4.pt),
                    "m4Eta":		    ak.flatten(ones * m4.eta),
                    "m4Phi":	        ak.flatten(ones * m4.phi)
                }
            }

            imap_concat = ak.concatenate([x[:, np.newaxis] for x in imap['features'].values()], axis=1)
            imap_scaled = (imap_concat - scaler.mean_) / scaler.scale_
            return (ak.values_astype(imap_scaled, "float32"),),{}
    if model_path is None:
        raise ValueError("NN Mass Reco Requires a model path.")
    model = jetAssignmentNN(model_path)
    outputs = model(events)
    m14 = jets[:, 0:4].sum().mass
    high_charg_score_mask = ak.unflatten(outputs[:,1] > 0.95, ak.num(jets))
    highest_3_charg_score_idx = ak.argsort(ak.unflatten(outputs[:,1], ak.num(jets)), axis=1)[:, -3:]
    highest_stop_score_idx = ak.argsort(ak.unflatten(outputs[:,0], ak.num(jets)), axis=1)[:, -1]
    top_3_charg_score_sum = jets[highest_3_charg_score_idx].sum()
    m3_top_3_nn_charg_score = top_3_charg_score_sum.mass
    m3_high_nn_charg_score = jets[high_charg_score_mask].sum().mass
    stop_jets = jets[ak.singletons(highest_stop_score_idx)]
    m4_nn = ak.flatten((top_3_charg_score_sum + stop_jets).mass)
    analyzer.H(
        f"{model_name}_m3_top_3_nn_charg_score",
        makeAxis(
            60,
            0,
            3000,
            rf"m3_top_3_nn_charg_score",
            unit="GeV",
        ),
        m3_top_3_nn_charg_score,
        description=f"Mass of sum of highest-scoring jets from {model_name}",
    )
    analyzer.H(
        f"{model_name}_m14_vs_m3_top_3_nn_charg_score",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, r"$m_{3 (NN)}$", unit="GeV"),
        ],
        [m14, m3_top_3_nn_charg_score],
        description=f"Masses of sum of highest-scoring jets from {model_name}",
    )

