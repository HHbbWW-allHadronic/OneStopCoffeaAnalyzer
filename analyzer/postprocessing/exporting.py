import functools as ft
import itertools as it

from .utils import gatherByPattern

from pydantic import BaseModel

# from .grouping import (
#     SectorGroupSpec,
#     createSectorGroups,
#     doFormatting,
#     SectorGroupParameters,
#     groupsMatch,
#     groupBy,
# )
from .grouping import doFormatting, SectorPipelineSpec
from .plots.export_hist import exportHist
from .registry import registerPostprocessor
from .processors import BasePostprocessor


@registerPostprocessor
class ExportHists(BasePostprocessor):
    histogram_names: list[str]
    input: SectorPipelineSpec
    output_name: str
    overwrite: bool = True

    def getNeededHistograms(self):
        return self.histogram_names

    def getFileFields(self):
        return set(self.input.group_fields.fields())

    def neededFileSets(self, params_mapping):
        return gatherByPattern(params_mapping, self.input.group_fields)

    def getExe(self, results):
        pipelines = self.input.makePipelines(results)
        for name, sector_pipeline in it.product(self.histogram_names, pipelines):
            histograms = sector_pipeline.getHists(name)
            provenance = histograms[0].provenance
            output_path = doFormatting(self.output_name, **provenance.allEntries())
            if len(histograms) != 1:
                raise RuntimeError()
            yield ft.partial(
                exportHist, histograms[0], output_path, overwrite=self.overwrite
            )

    def init(self):
        return
