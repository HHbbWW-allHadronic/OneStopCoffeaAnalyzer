from __future__ import annotations

import functools as ft
import itertools as it
from .plots.common import PlotConfiguration
from .style import StyleSet
from analyzer.utils.querying import BasePattern, Pattern, gatherByCapture, NO_MATCH
from analyzer.utils.structure_tools import (
    deepWalkMeta,
    SimpleCache,
    ItemWithMeta,
    commonDict,
    dictToDot,
    doFormatting
    
)
from .plots.plots_1d import plotOne
from .grouping2 import GroupBuilder
from analyzer.utils.structure_tools import globWithMeta
from attrs import define, field
import abc

ResultSet = list[list[ItemWithMeta]]

type PostprocessingGroup = dict[str,PostprocessingGroup] | list[PostprocessingGroup] | list[ItemWithMeta]


@define
class BasePostprocessor(abc.ABC):
    inputs: list[tuple[str,...]]
    structure: GroupBuilder


    def run(self, data):
        for i in self.inputs:
            items = globWithMeta(data, i)
            for x in self.structure.apply(items):
                yield from self.getRunFuncs(x) 

    @abc.abstractmethod
    def getRunFuncs(self, group: PostprocessingGroup):
        pass

@define
class Histogram1D(BasePostprocessor):
    output_name: str 
    style_set: str | StyleSet  = field(factory=StyleSet)
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    plot_configuration: PlotConfiguration = field(factory=PlotConfiguration)


    def getRunFuncs(self, group):
        
        if isinstance(group,dict):
            unstacked = group["unstacked"]
            stacked = group["stacked"]
        else:
            unstacked = group
            stacked = None
        common_meta = commonDict(it.chain((stacked or []),unstacked))
        output_path = doFormatting(self.output_name, **dict(dictToDot(common_meta)))
        pc = self.plot_configuration.makeFormatted(common_meta)
        print(unstacked)
        print(stacked)
        return
        yield ft.partial(plotOne, unstacked, stacked, common_meta, output_path,
                style_set=self.style_set,
                normalize=self.normalize,
                plot_configuration=pc,
                         )()
        # for name, sector_pipeline in it.product(self.histogram_names, pipelines):
        #     histograms = sector_pipeline.getHists(name)
        #     if not histograms:
        #         return
        #     provenance = histograms[0].provenance
        #     output_path = doFormatting(self.output_name, **provenance.allEntries())
        #     stacked_hists = None
        #     if self.to_stack is not None:
        #         stacked_hists = [
        #             x for x in histograms if self.to_stack.match(x.sector_parameters)
        #         ]
        #         histograms = [
        #             x
        #             for x in histograms
        #             if not self.to_stack.match(x.sector_parameters)
        #         ]
        # 
        #     pc = self.plot_configuration.makeFormatted(**provenance.allEntries())
        #     yield ft.partial(
        #         plotOne,
        #         histograms,
        #         provenance,
        #         output_path,
        #         scale=self.scale,
        #         style_set=self.style_set,
        #         normalize=self.normalize,
        #         plot_configuration=pc,
        #         stacked_hists=stacked_hists,
            # )
