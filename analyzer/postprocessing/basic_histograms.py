import functools as ft
import logging
import itertools as it
from typing import Literal
from pydantic import Field
from .grouping import SectorPipelineSpec, doFormatting, joinOnFields
from .plots.plots_1d import PlotConfiguration, plotOne, plotRatio
from .plots.plots_2d import plot2D
from .registry import registerPostprocessor
from .style import StyleSet
from .processors import BasePostprocessor
from analyzer.utils.querying import PatternExpression

logger = logging.getLogger(__name__)


@registerPostprocessor
class Histogram1D(BasePostprocessor):
    histogram_names: list[str]
    input: SectorPipelineSpec
    style_set: str | StyleSet
    output_name: str

    to_stack: PatternExpression | None = None

    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getNeededHistograms(self):
        return self.histogram_names

    def getFileFields(self):
        return set(self.input.group_fields.fields())

    def getExe(self, results):
        pipelines = self.input.makePipelines(results)

        for name, sector_pipeline in it.product(self.histogram_names, pipelines):
            histograms = sector_pipeline.getHists(name)
            if not histograms:
                return
            provenance = histograms[0].provenance
            output_path = doFormatting(self.output_name, **provenance.allEntries())
            stacked_hists = None
            if self.to_stack is not None:
                stacked_hists = [
                    x for x in histograms if self.to_stack.match(x.sector_parameters)
                ]
                histograms = [
                    x
                    for x in histograms
                    if not self.to_stack.match(x.sector_parameters)
                ]

            pc = self.plot_configuration.makeFormatted(**provenance.allEntries())
            yield ft.partial(
                plotOne,
                histograms,
                provenance,
                output_path,
                scale=self.scale,
                style_set=self.style_set,
                normalize=self.normalize,
                plot_configuration=pc,
                stacked_hists=stacked_hists,
            )


@registerPostprocessor
class Histogram2D(BasePostprocessor):

    histogram_names: list[str]
    style_set: str | StyleSet
    input: SectorPipelineSpec
    output_name: str = "{histogram_name}"
    normalize: bool = False
    color_scale: Literal["log", "linear"] = "linear"
    plot_configuration: PlotConfiguration | None = None

    def getNeededHistograms(self):
        return self.histogram_names

    def getFileFields(self):
        return set(self.input.group_fields.fields())

    def getExe(self, results):
        pipelines = self.input.makePipelines(results)
        for name, sector_pipeline in it.product(self.histogram_names, pipelines):
            for h in sector_pipeline.getHists(name):
                provenance = h.provenance
                output_path = doFormatting(self.output_name, **provenance.allEntries())
                self.plot_configuration.makeFormatted(**provenance.allEntries())
                output_path = doFormatting(self.output_name, **provenance.allEntries())
                yield ft.partial(
                    plot2D,
                    h,
                    output_path,
                    style_set=self.style_set,
                    normalize=self.normalize,
                    plot_configuration=self.plot_configuration,
                    color_scale=self.color_scale,
                )


@registerPostprocessor
class RatioPlot(BasePostprocessor):
    histogram_names: list[str]
    numerator: SectorPipelineSpec
    denominator: SectorPipelineSpec

    style_set: str | StyleSet

    output_name: str

    match_fields: list[str]

    plot_configuration: PlotConfiguration | None = None

    to_stack: PatternExpression | None = None

    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    ratio_ylim: tuple[float, float] = (0, 2)
    ratio_hlines: list[float] = Field(default_factory=lambda: [1.0])
    ratio_height: float = 0.5
    ratio_type: Literal["poisson", "poisson-ratio", "efficiency"] = "poisson"

    def getFileFields(self):
        return set(self.match_fields)

    def getNeededHistograms(self):
        return self.histogram_names

    def getExe(self, results):
        num_pipelines = self.numerator.makePipelines(results)
        den_pipelines = self.denominator.makePipelines(results)
        print(num_pipelines)
        print(den_pipelines)
        joined = joinOnFields(
            self.match_fields,
            num_pipelines,
            den_pipelines,
            key=lambda x: x.sector_group.field_values,
        )

        for name, (num, den) in it.product(self.histogram_names, joined):
            # if len(den) != 1:
            #     raise RuntimeError()
            den_hists = den[0].getHists(name)
            num_hists = [x for y in num for x in y.getHists(name)]
            # if len(den_hists) != 1:
            #     raise RuntimeError()
            den_hist = den_hists[0]
            output_path = doFormatting(
                self.output_name,
                **den_hist.provenance.allEntries(),
            )

            pc = self.plot_configuration.makeFormatted(
                **den_hist.provenance.allEntries()
            )
            yield ft.partial(
                plotRatio,
                den_hists,
                num_hists,
                output_path,
                self.style_set,
                normalize=self.normalize,
                ratio_ylim=self.ratio_ylim,
                ratio_type=self.ratio_type,
                scale=self.scale,
                ratio_hlines=self.ratio_hlines,
                ratio_height=self.ratio_height,
                plot_configuration=pc,
            )


#
# @registerPostprocessor
# class Histogram2DStack(BasePostprocessor, pyd.BaseModel):
#     histogram_names: list[str]
#     primary: SectorGroupSpec
#     signal: SectorGroupSpec
#     to_process: SectorSpec
#     style_set: str | StyleSet
#     output_name: str
#     match_fields: list[str]
#     scale: Literal["log", "linear"] = "linear"
#     match_fields: list[str]
#     axis_options: dict[str, Mode | str | int] | None = None
#     normalize: bool = False
#     override_axis_labels: dict[Literal["x", "y"], str] | None = None
#     plot_configuration: PlotConfiguration | None = None
#
#     def getNeededHistograms(self):
#         return self.histogram_names
#
#     def getExe(self, results):
#         print(self.plot_configuration)
#         results = [x for x in results if self.to_process.passes(x.sector_params)]
#         groups_primary = createSectorGroups(results, self.primary)
#         groups_signal = createSectorGroups(results, self.signal)
#         ret, items = [], []
#         for histogram in self.histogram_names:
#             for prim_group in groups_primary:
#                 if len(prim_group) != 1:
#                     raise KeyError(f"Too many groups")
#                 try:
#                     sig_group = list(
#                         x
#                         for x in groups_signal
#                         if groupsMatch(prim_group, x, self.match_fields)
#                     )
#                     if len(sig_group) != 1:
#                         raise KeyError(f"Too many groups")
#                     sig_group = next(iter(sig_group))
#                 except StopIteration:
#                     raise KeyError(f"Could not find group")
#
#                 bh = prim_group.histograms(histogram)
#                 sh = sig_group.histograms(histogram)
#                 if not bh or not sh:
#                     continue
#                 if len(bh) != 1 or len(sh) != 1:
#                     raise RuntimeError
#                 output = doFormatting(
#                     self.output_name,
#                     prim_group.all_parameters,
#                     histogram_name=histogram,
#                 )
#                 ret.append(
#                     ft.partial(
#                         plot2DSigBkg,
#                         bh[0],
#                         sh[0],
#                         output,
#                         self.style_set,
#                         normalize=self.normalize,
#                         plot_configuration=self.plot_configuration,
#                         override_axis_labels=self.override_axis_labels,
#                     )
#                 )
#                 items.append(
#                     PostprocessCatalogueEntry(
#                         processor_name=self.name,
#                         identifier=histogram,
#                         path=output,
#                         sector_group=prim_group,
#                         sector_params=[
#                             x.sector_params
#                             for x in [*prim_group.sectors, *sig_group.sectors]
#                         ],
#                     )
#                 )
#         return ret, items
