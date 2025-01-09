import abc
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

from .grouping import SectorGroupSpec, createSectorGroups, doFormatting
from .plots.export_hist import exportHist
from .plots.plots_1d import PlotConfiguration, plotOne, plotRatio, plotStrCat
from .plots.plots_2d import plot2D
from .registry import loadPostprocessors, registerPostprocessor
from .split_histogram import Mode
from .style import Style, StyleSet

logger = logging.getLogger(__name__)
StyleLike = Style | str


class BasePostprocessor(abc.ABC):
    @abc.abstractmethod
    def getExe(self, results):
        pass

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class Histogram1D(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    to_process: SectorSpec
    style_set: str | StyleSet
    output_name: str
    grouping: SectorGroupSpec

    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, self.grouping)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        plotOne,
                        sector_group.histograms(histogram),
                        sector_group.all_parameters,
                        doFormatting(
                            self.output_name,
                            sector_group.all_parameters,
                            histogram_name=histogram,
                        ),
                        scale=self.scale,
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret


@registerPostprocessor
class TriggerEff(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    to_process: SectorSpec
    style_set: str | StyleSet
    trigger_axis: str
    groupby: SectorGroupSpec
    output_name: str
    scale: Literal["log", "linear"] = "linear"

    axis_options: dict[str, Mode | str | int] | None = None
    normalize: bool = False

    plot_configuration: PlotConfiguration | None = None

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        plotOne,
                        histogram,
                        sector_group.parameters,
                        sector_group.sectors,
                        self.output_name,
                        scale=self.scale,
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret


@registerPostprocessor
class ExportHists(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    to_process: SectorSpec
    groupby: SectorGroupSpec
    output_name: str

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        exportHist,
                        histogram,
                        sector_group.parameters,
                        sector_group.sectors,
                        self.output_name,
                    )
                )
        return ret


@registerPostprocessor
class Histogram2D(BasePostprocessor, pyd.BaseModel):

    histogram_names: list[str]
    to_process: SectorSpec
    style_set: str | StyleSet

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

    axis_options: dict[str, Mode | str | int] | None = None
    normalize: bool = False

    color_scale: Literal["log", "linear"] = "linear"

    plot_configuration: PlotConfiguration | None = None

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        plot2D,
                        histogram,
                        sector_group.parameters,
                        sector_group.sectors[0],
                        self.output_name,
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                        color_scale=self.color_scale,
                    )
                )
        return ret


@registerPostprocessor
class PlotCutflow(BasePostprocessor, pyd.BaseModel):
    to_process: SectorSpec
    style_set: str | StyleSet
    groupby: SectorGroupSpec
    output_name: str
    normalize: bool = False
    table_mode: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for sector_group in r:
            ret.append(
                ft.partial(
                    plotStrCat,
                    sector_group.parameters,
                    sector_group.sectors,
                    self.output_name,
                    self.style_set,
                    table_mode=self.table_mode,
                    normalize=self.normalize,
                    plot_configuration=self.plot_configuration,
                )
            )
        return ret


@registerPostprocessor
class RatioPlot(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    numerator: SectorGroupSpec
    denominator: SectorGroupSpec
    # groupby: SectorGroupSpec
    style_set: str | StyleSet
    output_name: str
    scale: Literal["log", "linear"] = "linear"
    axis_options: dict[str, Mode | str | int] | None = None
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getExe(self, results):
        gnums = createSectorGroups(results, self.numerator)
        gdens = createSectorGroups(results, self.denominator)
        ret = []
        for histogram in self.histogram_names:
            for den_group in gdens:
                try:
                    num_group = next(x for x in gnums if den_group.compatible(x))
                except StopIteration:
                    raise KeyError(f"Could not find group {group}")

                dh = den_group.histograms(histogram)
                if len(dh) != 1:
                    raise RuntimeError
                ret.append(
                    ft.partial(
                        plotRatio,
                        dh[0],
                        num_group.histograms(histogram),
                        doFormatting(
                            self.output_name,
                            den_group.all_parameters,
                            histogram_name=histogram,
                        ),
                        self.style_set,
                        normalize=self.normalize,
                        scale=self.scale,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret
