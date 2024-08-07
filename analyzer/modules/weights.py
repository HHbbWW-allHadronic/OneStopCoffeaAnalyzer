import awkward as ak
from analyzer.core import analyzerModule, isMC
from coffea.analysis_tools import Weights


@analyzerModule(
    "pos_neg_weight", categories="post_selection", always=True, dataset_pred=isMC
)
def negEventWeight(events, analyzer):
    if "genWeight" in events.fields:
        analyzer.postsel_weights["PosNegEvent"] = {
            "central": ak.where(events.genWeight > 0, 1.0, -1.0)
        }
    return events, analyzer


@analyzerModule(
    "finalize_weights", categories="finalize_weights", always=True, after="weights"
)
def finalizeWeights(events, analyzer):
    analyzer.finalizeWeights()
    return events, analyzer


