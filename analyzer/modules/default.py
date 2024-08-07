import awkward as ak
from analyzer.core import analyzerModule


@analyzerModule("apply_selection", categories="apply_selection", always=True)
def applySelection(events, analyzer):
    events = analyzer.applySelection(events)
    return events, analyzer
