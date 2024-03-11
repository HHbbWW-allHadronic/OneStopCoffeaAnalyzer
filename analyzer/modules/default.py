import awkward as ak
from analyzer.core import analyzerModule


@analyzerModule("apply_selection", categories="post_selection", always=True)
def applySelection(events, analyzer):
    events = analyzer.applySelection(events)
    return events, analyzer
