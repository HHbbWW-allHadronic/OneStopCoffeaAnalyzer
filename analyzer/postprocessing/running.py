from cattrs import Converter
from analyzer.postprocessing.plots.common import PlotConfiguration
import concurrent.futures as cf
from analyzer.core.results import loadResults, mergeAndScale
from .processors import configureConverter
from .grouping import configureConverter as groupingConfConv
from .style import StyleSet
from analyzer.core.serialization import setupConverter
import matplotlib as mpl

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import yaml
from rich.progress import Progress, track
from distributed import (
    WorkerPlugin,
)
from analyzer.utils.querying import BasePattern
import analyzer.utils.querying
import analyzer.postprocessing.basic_histograms  # noqa
import analyzer.postprocessing.cutflows  # noqa
import analyzer.postprocessing.combine  # noqa
import analyzer.postprocessing.aggregate_plots  # noqa
import analyzer.postprocessing.exporting  # noqa
import analyzer.postprocessing.corrections  # noqa
from .style import loadStyles
from attrs import define, field
from rich import print
from .basic_histograms import BasePostprocessor


@define
class PostprocessorConfig:
    processors: list[BasePostprocessor]
    default_style_set: StyleSet = field(factory=StyleSet)
    default_plot_config: PlotConfiguration = field(factory=PlotConfiguration)
    drop_sample_pattern: BasePattern | None = None


def initProcess():
    mpl.use("Agg")
    loadStyles()


class LoadStyles(WorkerPlugin):
    def setup(self, worker):
        loadStyles()

    def teardown(self, worker):
        pass


def runPostprocessors(
    path, input_files, parallel=None, prefix=None, loaded_results=None
):
    converter = Converter()

    setupConverter(converter)
    groupingConfConv(converter)
    configureConverter(converter)

    loadStyles()

    with open(path, "r") as f:
        data = yaml.load(f, Loader=Loader)

    if "Postprocessing" in data:
        data = data["Postprocessing"]

    postprocessor = converter.structure(data, PostprocessorConfig)

    for processor in postprocessor.processors:
        if processor.style_set is None:
            processor.style_set = postprocessor.default_style_set
        if processor.plot_configuration is None:
            processor.plot_configuration = postprocessor.default_plot_config

    if loaded_results is not None:
        import copy

        results = copy.deepcopy(loaded_results)
    else:
        results = loadResults(input_files)

    results = mergeAndScale(
        results, drop_sample_pattern=postprocessor.drop_sample_pattern
    )
    all_funcs = []
    for processor in postprocessor.processors:
        all_funcs.extend(list(processor.run(results, prefix)))

    if parallel and parallel > 1:
        with Progress() as progress:
            task = progress.add_task("[green]Processing...", total=len(all_funcs))
            with cf.ProcessPoolExecutor(
                max_workers=parallel, initializer=initProcess
            ) as executor:
                futures = [executor.submit(f) for f in all_funcs]
                for future in cf.as_completed(futures):
                    future.result()
                    progress.update(task, advance=1)
    else:
        for f in track(all_funcs, description="Processing..."):
            f()
