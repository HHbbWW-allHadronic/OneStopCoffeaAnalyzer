import awkward as ak
from analyzer.core import analyzerModule

@analyzerModule("apply_selection", categories="post_selection", always=True)
def applySelection(events, analyzer):
    events = analyzer.applySelection(events)

    hlt_names = analyzer.profile.hlt
    hlt_names = ' | '.join(hlt_names)

    used_cuts = [i for i in analyzer.selection.names]

    analyzer.dask_result.set_cut_list(used_cuts)

    return events, analyzer
