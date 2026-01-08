from analyzer.cli.cli import postprocess
from debugpy import configure
from attr import Converter
from analyzer.postprocessing.plots.common import PlotConfiguration
import concurrent.futures as cf
from analyzer.core.results import loadResults, mergeAndScale
from cattrs.converters import Converter
from .processors import configureConverter
from .grouping import configureConverter as groupingConfConv
from .style import Style, StyleSet
from analyzer.core.serialization import setupConverter
import itertools as it
import matplotlib as mpl

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import yaml
from analyzer.utils.debugging import jumpIn
from collections import defaultdict
from rich import print
from rich.progress import Progress, track
from distributed import (
    Client,
    LocalCluster,
    fire_and_forget,
    get_client,
    secede,
    rejoin,
    Queue,
    as_completed,
    WorkerPlugin,
)
from analyzer.utils.querying import BasePattern
import analyzer.utils.querying
from .plots.mplstyles import loadStyles
import analyzer.postprocessing.basic_histograms  # noqa
from .plots.mplstyles import loadStyles
from attrs import field, define
from .basic_histograms import BasePostprocessor
from analyzer.utils.querying import BasePattern


@define
class PostprocessorConfig:
    processors: list[BasePostprocessor]
    default_style_set: StyleSet
    default_plot_config: PlotConfiguration
    drop_sample_patterns: list[BasePattern] | None = None


def initProcess():
    mpl.use("Agg")
    loadStyles()


class LoadStyles(WorkerPlugin):
    def setup(self, worker):
        loadStyles()

    def teardown(self, worker):
        pass


def runPostprocessors(path, input_files, parallel=8, prefix=None):
    converter = Converter()

    setupConverter(converter)
    groupingConfConv(converter)
    configureConverter(converter)

    loadStyles()

    with open(path, "r") as f:
        data = yaml.load(f, Loader=Loader)

    postprocessor = converter.structure(data, PostprocessorConfig)
    print(postprocessor.default_style_set)

    for processor in postprocessor.processors:
        if processor.style_set is None:
            processor.style_set = postprocessor.default_style_set
        if processor.plot_configuration is None:
            processor.plot_configuration = postprocessor.default_plot_config

    results = loadResults(input_files)
    results = mergeAndScale(results)
    all_funcs = []
    for processor in postprocessor.processors:
        all_funcs.extend(list(processor.run(results, prefix)))
    for f in all_funcs:
        f()


def main():
    from .transforms.registry import Transform

    runPostprocessors(
        "testpost.yaml",
        ["2026_01_07_no_jet_id_no_cleaning/*"],
    )


if __name__ == "__main__":
    main()
