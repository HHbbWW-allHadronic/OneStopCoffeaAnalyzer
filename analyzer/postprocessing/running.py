from cattrs import Converter
from analyzer.postprocessing.plots.common import PlotConfiguration
import concurrent.futures as cf
import math
from analyzer.core.results import loadResults, mergeAndScale

import heapq
from .processors import configureConverter as procConfConv
from .grouping import configureConverter as groupingConfConv
from .transforms.registry import configureConverter as transConfConv
from analyzer.utils.structure_tools import globWithMeta

from .style import StyleSet
from analyzer.core.serialization import setupConverter
import matplotlib as mpl

from rich.progress import Progress, track
from rich import print
from distributed import (
    WorkerPlugin,
)
from analyzer.utils.yamlload import loadTemplateYaml
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
    do_merge_and_scale: bool = True


def initProcess():
    mpl.use("Agg")
    loadStyles()


class LoadStyles(WorkerPlugin):
    def setup(self, worker):
        loadStyles()

    def teardown(self, worker):
        pass


def determineFileGroups(postprocessors, results):
    ret = set()
    for processor in postprocessors:
        real_inputs = [[("*", *inp) for inp in l] for l in processor.inputs]
        for i in real_inputs:
            items = [y for x in (globWithMeta(results, l) for l in i) for y in x]
            for x in processor.structure._applySimple(items):
                ret.add(frozenset(y.metadata["source_file"] for y in x))
    return ret


def makeApproxEqualSubgroups(groups, target_num_groups, size_func=lambda x: x):
    sets = [set() for _ in range(target_num_groups)]
    totals = [(0, i) for i in range(target_num_groups)]
    heapq.heapify(totals)
    for group in groups:
        total, index = heapq.heappop(totals)
        value = size_func(group)
        sets[index] |= group
        heapq.heappush(totals, (total + value, index))
    return sets


def maximalSubgroups(groups: set[frozenset]) -> set[frozenset]:
    ret = set()
    for x in groups:
        overlapped = [s for s in ret if not s.isdisjoint(x)]
        if not overlapped:
            ret.add(x)
            continue
        for s in overlapped:
            ret.discard(s)
        ret.add(frozenset(set().union(*overlapped, x)))
    return ret


def runPostprocessors(
    path,
    input_files,
    parallel=None,
    prefix=None,
    loaded_results=None,
    target_load_size: int | None = None,
    include_sidecar: bool = False,
):
    import analyzer.postprocessing.plots.utils as plots

    plots.INCLUDE_SIDECAR = include_sidecar

    converter = Converter()
    setupConverter(converter)

    transConfConv(converter)
    groupingConfConv(converter)
    procConfConv(converter)

    loadStyles()

    data = loadTemplateYaml(path)

    if "Postprocessing" in data:
        data = data["Postprocessing"]

    postprocessor = converter.structure(data, PostprocessorConfig)

    for processor in postprocessor.processors:
        if processor.style_set is None:
            processor.style_set = postprocessor.default_style_set
        if processor.plot_configuration is None:
            processor.plot_configuration = postprocessor.default_plot_config

    keep_patterns = []
    for processor in postprocessor.processors:
        if hasattr(processor, "inputs"):
            if postprocessor.do_merge_and_scale:
                keep_patterns.extend(
                    inp for inp_list in processor.inputs for inp in inp_list
                )
            else:
                keep_patterns.extend(
                    tuple(("*", *inp))
                    for inp_list in processor.inputs
                    for inp in inp_list
                )
    keep_patterns = keep_patterns or None

    if target_load_size is not None:
        peek_results, sizes = loadResults(
            input_files,
            keep_patterns=keep_patterns,
            peek_only=True,
            return_file_sizes=True,
        )
        file_groups = determineFileGroups(postprocessor.processors, peek_results)
    else:
        file_groups = [input_files]

    for file_group in file_groups:
        if loaded_results is not None:
            import copy

            results = copy.deepcopy(loaded_results)
        else:
            results = loadResults(file_group, keep_patterns=keep_patterns)

        if postprocessor.do_merge_and_scale:
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
