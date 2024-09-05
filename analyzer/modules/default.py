import awkward as ak
from analyzer.core import analyzerModule
from .utils import isData
from analyzer.core.lumi import getLumiMask
import logging


logger=logging.getLogger(__name__)

@analyzerModule("apply_selection", categories="apply_selection", always=True)
def applySelection(events, analyzer):
    events = analyzer.applySelection(events)
    hlt_names = analyzer.profile.hlt
    hlt_names = ' | '.join(hlt_names)

    used_cuts = [i for i in analyzer.selection.names]

    analyzer.dask_result.set_cut_list(used_cuts)

    return events, analyzer

@analyzerModule("golden_json_filter", categories="init", always=True, dataset_pred=isData)
def goldenJsonFilter(events, analyzer):
    return events, analyzer
    profile = analyzer.profile
    lumi_json = profile.lumi_json
    lmask = getLumiMask(lumi_json)
    events = events[lmask(events.run, events.luminosityBlock)]
    return events, analyzer
