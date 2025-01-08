import functools as ft
import itertools as it
import logging
from pathlib import Path
from typing import Literal

import yaml

import pydantic as pyd
from analyzer.configuration import CONFIG
from analyzer.core.results import loadSampleResultFromPaths, makeDatasetResults
from analyzer.core.specifiers import SectorSpec
from rich import print


def quicklookSample(result):
    data = {
        "sample_name": result.sample_id.sample_name,
        "data_name": result.sample_id.dataset_name,
        "params": result.params,
        "processed_events": result.processed_events,
        "expected_events": result.params.n_events,
        "regions": list(result.results),
        "region_hists": {
            x: list(y.base_result.histograms) for x, y in result.results.items()
        },
    }
    print(data)


def quicklookHist(result, region, hist):
    h = result.results[region].base_result.histograms[hist].histogram
    print(h)
    return h


def quicklookFiles(paths):
    results = loadSampleResultFromPaths(paths)
    for k, v in results.items():
        quicklookSample(v)


def quicklookHistsPath(paths, region, hist, interact=False):
    results = loadSampleResultFromPaths(paths)
    for k, v in results.items():
        histogram = quicklookHist(v, region, hist)
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
