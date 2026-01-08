import functools as ft
from typing import Literal


from pydantic import BaseModel
from .misc import dumpYield

# # from .grouping import (
# #     SectorGroupSpec,
# #     createSectorGroups,
# #     doFormatting,
# #     SectorGroupParameters,
# #     groupsMatch,
# #     groupBy,
# # )
# from .grouping import SectorPipelineSpec, doFormatting
# from .plots.plots_1d import PlotConfiguration, plotStrCat
# from .registry import registerPostprocessor
# from .style import StyleSet
# from .processors import BasePostprocessor


@registerPostprocessor
class PlotCutflow(BasePostprocessor, BaseModel):
    input: SectorPipelineSpec

    style_set: str | StyleSet

    output_name: str

    plot_types: list[str] = ["cutflow", "one_cut", "n_minus_one"]
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    table_mode: bool = False
    weighted: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getFileFields(self):
        return set(self.input.group_fields.fields())

    def getExe(self, results):
        pipelines = self.input.makePipelines(results)
        for sector_pipeline in pipelines:
            for pt in self.plot_types:
                pc = self.plot_configuration.makeFormatted(
                    **sector_pipeline.sector_group.field_values
                )
                output = doFormatting(
                    self.output_name,
                    **sector_pipeline.sector_group.field_values,
                    histogram_name=pt,
                )
                yield ft.partial(
                    plotStrCat,
                    pt,
                    sector_pipeline.sector_group.sectors,
                    output,
                    self.style_set,
                    table_mode=self.table_mode,
                    normalize=self.normalize,
                    plot_configuration=pc,
                    scale=self.scale,
                )


@registerPostprocessor
class DumpYields(BasePostprocessor, BaseModel):
    input: SectorPipelineSpec
    output_name: str

    def getFileFields(self):
        return set(self.input.group_fields.fields())

    def getExe(self, results):
        pipelines = self.input.makePipelines(results)
        for sector_pipeline in pipelines:
            output = doFormatting(
                self.output_name,
                **sector_pipeline.sector_group.field_values,
            )
            yield ft.partial(dumpYield, sector_pipeline.sector_group.sectors, output)
