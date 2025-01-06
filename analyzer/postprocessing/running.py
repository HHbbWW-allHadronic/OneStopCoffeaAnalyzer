from analyzer.logging import setup_logging
from .plots.mplstyles import loadStyles
import mplhep as hep
import functools as ft
import itertools as it
import logging
from rich import print
from pathlib import Path
from typing import Literal

import yaml

import pydantic as pyd
from analyzer.configuration import CONFIG

from analyzer.core.specifiers import SectorSpec
from rich.progress import Progress
import concurrent.futures as cf
import matplotlib as mpl


from .plots.export_hist import exportHist
from .plots.plots_1d import PlotConfiguration, plotOne, plotRatio, plotStrCat
from .plots.plots_2d import plot2D
from .registry import loadPostprocessors, registerPostprocessor
from .split_histogram import Mode
from .style import Style, StyleSet
from .grouping import createSectorGroups
import analyzer.postprocessing.processors

from analyzer.core.results import loadSampleResultFromPaths, makeDatasetResults


def runPostprocessors(config, input_files, parallel=8):
    loadStyles()
    print("Loading Postprocessors")
    loaded = loadPostprocessors("configurations/post.yaml")
    print("Loading Samples")
    sample_results = loadSampleResultFromPaths(input_files)
    dataset_results = makeDatasetResults(sample_results)
    print("Ready to Process ")
    sector_results = list(
        it.chain.from_iterable(r.sector_results for r in dataset_results.values())
    )


    tasks = []

    for processor in loaded:
        processor.init()
        tasks += processor.getExe(sector_results)


    
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Processing...", total=len(tasks))
        if not parallel:
            for f in tasks:
                f()
                progress.advance(task_id)
        else:
            with cf.ProcessPoolExecutor(max_workers=parallel) as executor:
                results = [executor.submit(f) for f in tasks]
                for i in cf.as_completed(results):
                    progress.advance(task_id)
