import functools as ft
import itertools as it
import logging
from pathlib import Path
from typing import Literal

import yaml

import pydantic as pyd
from analyzer.configuration import CONFIG

from analyzer.core.specifiers import SectorSpec


from .plots.export_hist import exportHist
from .plots.plots_1d import PlotConfiguration, plotOne, plotRatio, plotStrCat
from .plots.plots_2d import plot2D
from .registry import loadPostprocessors, registerPostprocessor
from .split_histogram import Mode
from .style import Style, StyleSet
from .grouping import createSectorGroups

from analyzer.core.results import loadSampleResultFromPaths, makeDatasetResults

logger = logging.getLogger(__name__)
StyleLike = Style | str


@registerPostprocessor
class Histogram1D(pyd.BaseModel):
    histogram_names: list[str]

    to_process: SectorSpec

    style_set: str | StyleSet

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

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
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class ExportHists(pyd.BaseModel):
    histogram_names: list[str]
    to_process: SectorSpec
    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

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

    def init(self):
        pass


@registerPostprocessor
class Histogram2D(pyd.BaseModel):

    histogram_names: list[str]
    to_process: SectorSpec
    style_set: str| StyleSet

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

    axis_options: dict[str, Mode| str| int] | None = None
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

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class PlotCutflow(pyd.BaseModel):
    to_process: SectorSpec
    style_set: str| StyleSet

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"
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

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class RatioPlot(pyd.BaseModel):
    histogram_names: list[str]
    numerator: SectorSpec
    denominator: SectorSpec
    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    style_set: str| StyleSet
    output_name: str = "{histogram_name}"
    axis_options: dict[str, Mode | str| int] | None = None
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getExe(self, results):
        nums = [x for x in results if self.numerator.passes(x.sector_params)]
        dens = [x for x in results if self.denominator.passes(x.sector_params)]
        gnums = createSectorGroups(nums, *self.groupby)
        gdens = createSectorGroups(dens, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for den_group in gdens:
                try:
                    num_group = next(x for x in gnums if den_group.compatible(x))
                except StopIteration:
                    raise KeyError(f"Could not find group {group}")

                ret.append(
                    ft.partial(
                        plotRatio,
                        histogram,
                        sector_group.parameters,
                        num_group.sectors,
                        den_group.sectors[0],
                        self.output_name,
                        self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


if __name__ == "__main__":
    from analyzer.logging import setup_logging
    from .plots.mplstyles import loadStyles

    loadStyles()
    setup_logging()

    loaded = loadPostprocessors("configurations/post.yaml")
    # result = AnalysisResult.fromFile("results/histograms/2024_10_06.pkl")
    # result = AnalysisResult.fromFile("results/histograms/2024_11_05_patched.pkl")
    # result = AnalysisResult.fromFile("test.pkl")
    sample_results = loadSampleResultFromPaths(Path("test_results").glob("*.pkl"))
    dataset_results = makeDatasetResults(sample_results)
    sector_results = list(
        it.chain.from_iterable(r.sector_results for r in dataset_results.values())
    )

    tasks = []

    for processor in loaded:
        processor.init()
        tasks += processor.getExe(sector_results)

    results = [f() for f in tasks]

    import sys

    sys.exit()
    print("HERE")
    # with Progress() as progress:
    #     task_id = progress.add_task("[cyan]Processing...", total=len(tasks))
    #     with cf.ProcessPoolExecutor(max_workers=8) as executor:
    #         results = [executor.submit(f) for f in tasks]
    #         for i in cf.as_completed(results):
    #             progress.advance(task_id)
