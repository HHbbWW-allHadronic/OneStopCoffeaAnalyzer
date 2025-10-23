from analyzer.core.results import loadSampleResultFromPaths
from rich import print, inspect
import hist


def quicklookSample(result, include_hists=False, include_other=False):
    res = result.results or {}
    data = {
        "sample_name": result.sample_id.sample_name,
        "data_name": result.sample_id.dataset_name,
        # "params": result.params,
        "processed_events": result.processed_events,
        "expected_events": result.params.n_events,
    }

    if include_hists:
        data["region_hists"] = {
            x: list(y.base_result.histograms) for x, y in res.items()
        }
    else:
        data["region_hists"] = {
            x: len(y.base_result.histograms) for x, y in res.items()
        }
    if include_other:
        data["region_other"] = {
            x: list(y.base_result.other_data) for x, y in res.items()
        }
    else:
        data["region_other"] = {
            x: len(y.base_result.other_data) for x, y in res.items()
        }

    data.update(
        {
            "region_cutflow": {
                x: list(y.base_result.selection_flow.cutflow) for x, y in res.items()
            },
            "weight_flow": {
                x: y.base_result.post_sel_weight_flow for x, y in res.items()
            },
            "pre_sel_weight_flow": {
                x: y.base_result.pre_sel_weight_flow for x, y in res.items()
            },
        }
    )
    print(data)

def quicklookHist(result, region, hist_name, variation=None, rebin=None):
    h = result.results[region].base_result.histograms[hist_name].histogram
    if variation:
        h = h[{"variation" : variation}]
    if rebin:
        h = h[hist.rebin(rebin)]
    print(h)
    return h


def quicklookFiles(paths,**kwargs):
    print(kwargs)
    results = loadSampleResultFromPaths(paths,decompress=True,show_progress=True)
    for k, v in results.items():
        quicklookSample(v,**kwargs)


def quicklookHistsPath(paths, region, hist_name, interact=False, variation=None, rebin=None):
    results = loadSampleResultFromPaths(paths)
    for k, v in results.items():
        histogram = quicklookHist(v, region, hist_name, variation=variation, rebin=rebin)
    if interact:
        import code
        import readline
        import rlcompleter

        vars = globals()
        vars.update(locals())
        # vars = {"histogram": h}
        readline.set_completer(rlcompleter.Completer(vars).complete)
        readline.parse_and_bind("tab: complete")
        code.InteractiveConsole(vars).interact()
