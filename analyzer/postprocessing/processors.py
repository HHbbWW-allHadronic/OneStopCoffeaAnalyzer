import concurrent.futures as cf
import functools as ft
import logging
from pathlib import Path
from typing import Optional, Union

import pydantic as pyd
import yaml
from analyzer.configuration import CONFIG
from analyzer.core import AnalysisResult
from analyzer.core.specifiers import SectorSpec
from rich.progress import track

from .plots.plots_1d import PlotConfiguration, plotOne, plotRatio, plotStrCat
from .plots.plots_2d import plot2D
from .registry import loadPostprocessors, registerPostprocessor
from .split_histogram import Mode
from .style import Style, StyleSet
from .utils import createSectorGroups

logger = logging.getLogger(__name__)
StyleLike = Union[Style, str]


@registerPostprocessor
class Histogram1D(pyd.BaseModel):

    histogram_names: list[str]
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

    axis_options: Optional[dict[str, Union[Mode, str, int]]] = None
    normalize: bool = False

    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        plotOne,
                        histogram,
                        sector_group.sectors,
                        self.output_name,
                        self.style_set,
                        self.normalize,
                        self.plot_configuration,
                    )
                )
        return ret

    def loadStyle(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class Histogram2D(pyd.BaseModel):

    histogram_names: list[str]
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

    axis_options: Optional[dict[str, Union[Mode, str, int]]] = None
    normalize: bool = False

    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        plot2D,
                        histogram,
                        sector_group.sectors[0],
                        self.output_name,
                        self.style_set,
                        self.normalize,
                        self.plot_configuration,
                    )
                )
        return ret

    def loadStyle(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class PlotCutflow(pyd.BaseModel):
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"
    normalize: bool = False
    table_mode: bool = False
    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for sector_group in r:
            ret.append(
                ft.partial(
                    plotStrCat,
                    sector_group.sectors,
                    self.output_name,
                    self.style_set,
                    table_mode=self.table_mode,
                    normalize=self.normalize,
                    plot_configuration=self.plot_configuration,
                )
            )
        return ret

    def loadStyle(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class DatasetRatioPlot(pyd.BaseModel):
    histogram_names: list[str]
    numerator: SectorSpec
    denominator: SectorSpec
    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    style_set: Union[str, StyleSet]
    output_name: str = "{histogram_name}"
    axis_options: Optional[dict[str, Union[Mode, str, int]]] = None
    normalize: bool = False
    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        nums = [x for x in results.values() if self.numerator.passes(x.sector_params)]
        dens = [x for x in results.values() if self.denominator.passes(x.sector_params)]
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
                        num_group.sectors,
                        den_group.sectors[0],
                        self.output_name,
                        self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret

    def loadStyle(self):
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
    result = AnalysisResult.fromFile("results/histograms/2024_10_01_v2.pkl")
    result = result.getResults()
    tasks = []
    for processor in loaded:
        processor.loadStyle()
        tasks += processor.getExe(result)

    # results = [f() for f in tasks]

    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        results = [executor.submit(f) for f in tasks]
        for i in track(cf.as_completed(results), total=len(results)):
            i.result()

    # # mplhep.style.use("CMS")

    # config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
    # with open(config_path, "r") as f:
    #     d = yaml.safe_load(f)
    # ss = StyleSet(**d)

    # result = AnalysisResult.fromFile("test2.pkl")
    # result = result.getResults()
    # h1 = Histogram1D(
    #     histogram_names=["HT", "m13_m", "h_njet", "m14_m"],
    #     groupby=["dataset.name", "dataset.era.name", "region.region_name"],
    #     output_name="{dataset.era.name}/{region.region_name}/{dataset.name}/{histogram_name}.pdf",
    #     to_plot=SectorSpec(sample_spec=SampleSpec(name="signal_312_*")),
    #     style_set=ss,
    #     normalize=False,
    #     plot_configuration=PlotConfiguration(y_label="Normalized Events"),
    # )
    # h2 = RatioPlot(
    #     histogram_names=["HT"],
    #     groupby=["dataset.era.name", "region.region_name"],
    #     output_name="{dataset.era.name}/{region.region_name}/{dataset.name}/ratio_{histogram_name}.pdf",
    #     numerator=SectorSpec(sample_spec=SampleSpec(name="signal_312_15*")),
    #     denominator=SectorSpec(sample_spec=SampleSpec(name="signal_312_2000_1900")),
    #     style_set=ss,
    #     normalize=True,
    #     plot_configuration=PlotConfiguration(y_label="Normalized Events"),
    # )
    # r1 = h1.getExe(result)
    # r2 = h2.getExe(result)
    # # [f() for f in it.chain(r2)]
    # with cf.ProcessPoolExecutor(max_workers=8) as executor:
    #     results = [executor.submit(f) for f in it.chain(r2, r1)]
    #     for i in track(cf.as_completed(results), total=len(results)):
    #         i.result()
