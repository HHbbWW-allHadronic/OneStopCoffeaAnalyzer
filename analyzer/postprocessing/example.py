import concurrent.futures as cf
import functools as ft
import itertools as it
import logging
import operator as op
import string
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pydantic as pyd
import yaml
from analyzer.configuration import CONFIG
from analyzer.core.specifiers import SampleSpec, SectorSpec
from rich import print

from .plotting import (
    autoScale,
    drawAs1DHist,
    drawAsScatter,
    drawRatio,
    getRatioAndUnc,
    labelAxis,
    loadStyles,
)
from .plotting.utils import addAxesToHist
from .split_histogram import Mode
from .style import Style, Styler, StyleSet

logger = logging.getLogger(__name__)
StyleLike = Union[Style, str]


def getNested(d, s):
    parts = s.split(".")

    def getK(di, p):
        return di[p]

    ret = ft.reduce(getK, parts, d)
    return ret


def doFormatting(s, sector_params, **kwargs):
    parsed = string.Formatter().parse(s)
    d = sector_params.model_dump()
    s = ""
    for x in parsed:
        s += x[0]
        if x[1] is not None:
            if x[1] in kwargs:
                s += kwargs[x[1]]
            else:
                s += getNested(d, x[1])
    return s


def groupBy(fields, data):
    def k(v):
        return tuple([getNested(v.sector_params.model_dump(), x) for x in fields])

    grouped = it.groupby(sorted(data, key=k), k)
    ret = [(dict(zip(fields, x)), list(y)) for x, y in grouped]
    return ret


def saveFig(fig, out, extension=".pdf", metadata=None):

    path = Path(out)
    path.parent.mkdir(exist_ok=True, parents=True)
    path = path.with_suffix(extension)
    fig.savefig(path,metadata=metadata)


def addCMSBits(ax, sectors):
    lumis = set(str(x.sector_params.dataset.lumi) for x in sectors)
    energies = set(str(x.sector_params.dataset.era.energy) for x in sectors)
    lumi_text = f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
    mplhep.cms.lumitext(text=lumi_text, ax=ax)
    mplhep.cms.text(text="Preliminary", ax=ax)


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
        drawAs1DHist(
            ax,
            h.axes[0],
            h.values(),
            y_unc=np.sqrt(h.variances()),
            title=sector.sector_params.dataset.title,
            style=style,
        )
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    autoScale(ax)
    addCMSBits(ax, sectors)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    saveFig(fig, Path("plots") / o)
    plt.close(fig)


def _plotRatio(
    histogram,
    group,
    numerators,
    denominator,
    output_name,
    style_set,
    normalize=False,
    middle=1,
    y_lim=None,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    den = denominator.histograms[histogram].get()
    addAxesToHist(ax, num_bottom=1)
    drawAs1DHist(
        ax,
        den.axes[0],
        den.values(),
        y_unc=np.sqrt(den.variances()),
        title=denominator.sector_params.dataset.title,
        style=styler.getStyle(denominator.sector_params),
    )
    all_r, all_u = [], []
    for sector in numerators:
        p = sector.sector_params
        style = styler.getStyle(p)
        h = sector.histograms[histogram].get()
        ratio, unc = getRatioAndUnc(h.values(), den.values())
        all_r.append(ratio)
        all_u.append(unc)
        s = styler.getStyle(sector.sector_params)

        drawAs1DHist(
            ax,
            h.axes[0],
            h.values(),
            y_unc=np.sqrt(h.variances()),
            title=sector.sector_params.dataset.title,
            style=s,
        )
        drawAsScatter(ax.bottom_axes[0], h.axes[0], ratio, y_unc=unc, style=s)
    all_r = np.concatenate(all_r).flatten()
    all_u = np.concatenate(all_u, axis=1)
    if y_lim is None:
        valid_ratios_idx = np.where(~np.isnan(all_r))
        valid_ratios = all_r[valid_ratios_idx]
        extrema = np.array(
            [
                valid_ratios - all_u[0][valid_ratios_idx],
                valid_ratios + all_u[1][valid_ratios_idx],
            ]
        )
        max_delta = np.amax(np.abs(extrema - middle))
        ratio_extrema = np.abs(max_delta + middle)
        _alpha = 2.0
        scaled_offset = max_delta + (max_delta / (_alpha * ratio_extrema))
        y_lim = [middle - scaled_offset, middle + scaled_offset]

    labelAxis(ax, "y", den.axes)
    labelAxis(ax.bottom_axes[0], "x", den.axes)
    autoScale(ax)
    addCMSBits(ax, (denominator,))
    ax.bottom_axes[0].set_ylim([0, 2])
    ax.bottom_axes[0].set_ylabel("Ratio")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    o = doFormatting(output_name, p, histogram_name=histogram)
    fig.tight_layout()
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


class RatioPlot(pyd.BaseModel):

    histogram_names: list[str]
    numerator: SectorSpec
    denominator: SectorSpec
    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    style_set: StyleSet
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
                n = next(x for x in gnums if x[0] == group)
                ret.append(
                    ft.partial(
                        _plotRatio,
                        histogram,
                        group,
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
    h1 = Histogram1D(
        histogram_names=["HT", "m13_m", "h_njet", "m14_m"],
        groupby=["dataset.name", "dataset.era.name", "region.region_name"],
        output_name="{dataset.era.name}/{region.region_name}/{dataset.name}/{histogram_name}.pdf",
        to_plot=SectorSpec(sample_spec=SampleSpec(name="signal_312_*")),
        style_set=ss,
        normalize=False,
        plot_configuration=PlotConfiguration(y_label="Normalized Events"),
    )
    h2 = RatioPlot(
        histogram_names=["HT"],
        groupby=["dataset.era.name", "region.region_name"],
        output_name="{dataset.era.name}/{region.region_name}/{dataset.name}/ratio_{histogram_name}.pdf",
        numerator=SectorSpec(sample_spec=SampleSpec(name="signal_312_15*")),
        denominator=SectorSpec(sample_spec=SampleSpec(name="signal_312_2000_1900")),
        style_set=ss,
        normalize=True,
        plot_configuration=PlotConfiguration(y_label="Normalized Events"),
    )
    r1 = h1.getExe(result)
    r2 = h2.getExe(result)
    # [f() for f in it.chain(r2)]
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        results = [executor.submit(f) for f in it.chain(r2, r1)]
        for i in track(cf.as_completed(results), total=len(results)):
            i.result()
