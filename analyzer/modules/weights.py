import awkward as ak
from analyzer.core import analyzerModule
from coffea.analysis_tools import Weights


@analyzerModule("weights", categories="weights", always=True)
def addWeights(events, analyzer):
    if analyzer.delayed:
        analyzer.weights = Weights(None)
    else:
        analyzer.weights = Weights(ak.num(events, axis=0))

    if "genWeight" in events.fields:
        analyzer.weights.add("PosNeg", ak.where(events.genWeight > 0, 1.0, -1.0))
    return events, analyzer
