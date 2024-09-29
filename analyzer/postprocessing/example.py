import concurrent.futures as cf
import functools as ft
import itertools as it
import logging
import operator as op
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import mplhep
import pydantic as pyd
import yaml
from analyzer.configuration import CONFIG
from analyzer.core.specifiers import SampleSpec, SectorSpec
from rich import print

from .plotting import drawAs1DHist, loadStyles, setAxisTitles1D
from .split_histogram import Mode
from .style import Style, Styler, StyleSet


logger = logging.getLogger(__name__)


StyleLike = Union[Style, str]


class BasePostprocessor(pyd.BaseModel):
    output_dir: str
    styles: list[Union[Style, str]]

    def resolveStyle(self, dataset, region):
        pass


def getNested(d, s):
    parts = s.split(".")
    def getK(di, p):
        return di[p]
    ret = ft.reduce(getK, parts, d)
    return ret


def groupBy(fields, data):
    # Group = namedtuple("Group", [x.replace(".", "_") for x in fields])

    def k(v):
        return tuple([getNested(v.sector_params.model_dump(), x) for x in fields])

    grouped = it.groupby(sorted(data, key=k), k)
    ret = [(dict(zip(fields, x)), list(y)) for x, y in grouped]
    return ret


def saveFig(fig, out, extension=".pdf"):
    path = Path(out)
    path.parent.mkdir(exist_ok=True, parents=True)
    path = path.with_suffix(extension)
    fig.savefig(path)


def addCMSBits(ax, sectors):
    lumis = set(str(x.sector_params.dataset.lumi) for x in sectors)
    energies = set(
        str(x.sector_params.dataset.era.energy) for x in sectors
    )
    lumi_text = f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
    mplhep.cms.lumitext(text=lumi_text, ax=ax)
    mplhep.cms.text(text="Preliminary", ax=ax)
    mplhep.sort_legend(ax=ax)


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


def _plotOne(
    histogram,
    group,
    sectors,
    output_name,
    style_set,
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    for sector in sectors:
        p = sector.sector_params
        style = styler.getStyle(p)
        h = sector.histograms[histogram].get()
        if normalize:
            h = h / h.sum().value
        drawAs1DHist(ax, h, title=sector.sector_params.dataset.title, style=style)
    setAxisTitles1D(
        ax,
        h.axes[0],
        x_label=pc.x_label,
        y_label=pc.y_label,
        y_label_complete=pc.y_label_complete,
    )
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    t = "_".join(group.values())
    o = output_name.format(histogram_name=histogram, **group)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


class Histogram1D(pyd.BaseModel):

    histogram_names: list[str]
    to_plot: SectorSpec
    style_set: StyleSet

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
                        group,
                        sector,
                        self.output_name,
                        self.style_set,
                        self.normalize,
                        self.plot_configuration,
                    )
                )
        return ret


class RatioPlot(BasePostprocessor):
    name_format: str
    numerator: list[SectorSpec]
    denominator: list[SectorSpec]

    def __init__(self, default_configuration):
        pass

    def getExe(self, dataset_results):
        pass


if __name__ == "__main__":
    from analyzer.core import AnalysisResult
    from analyzer.logging import setup_logging
    from rich.progress import track

    setup_logging()

    loadStyles()
    # mplhep.style.use("CMS")

    config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
    with open(config_path, "r") as f:
        d = yaml.safe_load(f)
    ss = StyleSet(**d)

    result = AnalysisResult.fromFile("test.pkl")
    result = result.getResults()
    h = Histogram1D(
        histogram_names=["HT"],
        to_plot=SectorSpec(sample_spec=SampleSpec(name="signal_312_200*")),
        style_set=ss,
        normalize=True,
        plot_configuration=PlotConfiguration(y_label="Normalized Events"),
    )
    r = h.getExe(result)
    with cf.ProcessPoolExecutor(max_workers=4) as executor:
        results = [executor.submit(f) for f in r]
        for i in track(cf.as_completed(results), total=len(results)):
            i.result()
