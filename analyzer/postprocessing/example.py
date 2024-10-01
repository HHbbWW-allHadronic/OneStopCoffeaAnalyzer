import functools as ft
import logging
from typing import Optional, Union


import pydantic as pyd
from analyzer.core.specifiers import SectorSpec
from rich import print

from .registry import loadPostprocessors, registerPostprocessor
from .split_histogram import Mode
from .style import Style, StyleSet

logger = logging.getLogger(__name__)
StyleLike = Union[Style, str]





class PlotConfiguration(pyd.BaseModel):
    lumi_str: Optional[str] = None
    era_str: Optional[str] = None
    energy_str: Optional[str] = None
    extra_text: Optional[str] = None
    cms_text: Optional[str] = None

    x_scale: Optional[str] = "linear"
    y_scale: Optional[str] = "linear"

    x_label: Optional[str] = None
    y_label: Optional[str] = None
    y_label_complete: Optional[str] = None


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
        r = groupBy(self.groupby, sectors)
        ret = []
        for histogram in self.histogram_names:
            for group, sector in r:
                ret.append(
                    ft.partial(
                        _plotOne,
                        histogram,
                        sector,
                        self.output_name,
                        self.style_set,
                        self.normalize,
                        self.plot_configuration,
                    )
                )
        return ret


@registerPostprocessor
class PlotCutflow(pyd.BaseModel):
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"
    normalize: bool = False
    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = groupBy(self.groupby, sectors)
        ret = []
        for histogram in self.histogram_names:
            for group, sector in r:
                ret.append(
                    ft.partial(
                        _plotOne,
                        histogram,
                        sector,
                        self.output_name,
                        self.style_set,
                        self.normalize,
                        self.plot_configuration,
                    )
                )
        return ret


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
        gnums = groupBy(self.groupby, nums)
        gdens = groupBy(self.groupby, dens)
        ret = []
        for histogram in self.histogram_names:
            for group, d in gdens:
                try:
                    n = next(x for x in gnums if x[0] == group)
                except StopIteration:
                    raise KeyError(f"Could not find group {group}")

                ret.append(
                    ft.partial(
                        _plotRatio,
                        histogram,
                        n[1],
                        d[0],
                        self.output_name,
                        self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret


if __name__ == "__main__":
    from analyzer.logging import setup_logging

    setup_logging()

    x = loadPostprocessors("configurations/post.yaml")
    print(x)

    # loadStyles()
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
