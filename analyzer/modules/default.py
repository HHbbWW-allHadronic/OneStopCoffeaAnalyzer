import awkward as ak
from analyzer.core import analyzerModule
from .utils import isData
from analyzer.core.lumi import getLumiMask


@analyzerModule("apply_selection", categories="apply_selection", always=True)
def applySelection(events, analyzer):
    events = analyzer.applySelection(events)
    return events, analyzer

@analyzerModule("golden_json_filter", categories="init", always=True, dataset_pred=isData)
def applySelection(events, analyzer):
    profile = analyzer.profile
    lumi_json = profile.lumi_json
    lmask = getLumiMask(lumi_json)
    events = events[lmask(events.run, events.luminosityBlock)]
    return events, analyzer
