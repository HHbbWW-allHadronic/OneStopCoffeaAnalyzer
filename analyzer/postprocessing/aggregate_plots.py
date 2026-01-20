from __future__ import annotations

import functools as ft
from typing import Literal
import itertools as it
from .style import StyleSet, Style
from analyzer.utils.structure_tools import (
    ItemWithMeta,
    commonDict,
    dictToDot,
    doFormatting,
)
from .processors import BasePostprocessor
from .plots.plots_1d import plotOne, plotRatio
from .plots.plots_2d import plot2D
from analyzer.utils.querying import BasePattern, deepLookup
from attrs import define, field
import numpy as np
import enum

ResultSet = list[list[ItemWithMeta]]


class SignificanceType(str, enum.Enum):
    poisson_basic = "poisson_basic"
    poisson_low_stat = "poisson_low_stat"


class SignificanceCalculation(str, enum.Enum):
    single_bin = "single_bin"
    quadrature_sum = "quadrature_sum"


def poisson_basic(s, b):
    return s / np.sqrt(b)


def poisson_low_stat(s, b):
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))


def single_bin(f, s, b):
    return f(np.sum(s), np.sum(b))


def quadrature_sum(f, s, b):
    mask = b > 0.001
    s, b = s[mask], b[mask]

    return np.sqrt(np.sum(np.square(f(s, b))))


significance_type_funcs = {
    SignificanceType.poisson_basic: poisson_basic,
    SignificanceType.poisson_low_stat: poisson_low_stat,
}

significance_calculation_funcs = {
    SignificanceCalculation.single_bin: single_bin,
    SignificanceCalculation.quadrature_sum: quadrature_sum,
}


def makeSignificance2D(
    signal,
    background,
    common_metadata,
    output_path,
    significance_type,
    significance_calculation,
    xy_pattern,
    xyz_labels,
    style,
    plot_configuration=None,
    **kwargs,
):
    import matplotlib.pyplot as plt
    from .plots.annotations import addCMSBits, labelAxis
    from .plots.common import PlotConfiguration
    from .plots.utils import saveFig

    background_hist = background.item.histogram
    sigs = []
    for item, meta in signal:
        h = item.histogram
        xy = (
            float(deepLookup(meta, xy_pattern[0])),
            float(deepLookup(meta, xy_pattern[1])),
        )
        sig = significance_calculation_funcs[significance_calculation](
            significance_type_funcs[significance_type],
            h.values(),
            background_hist.values(),
        )
        sigs.append((*xy, sig))
    sigs = np.array(sigs)

    fig, ax = plt.subplots()

    sc = ax.scatter(
        sigs[:, 0],
        sigs[:, 1],
        c=sigs[:, 2],
        **style.get("scatter_z", include_type=False),
    )
    fig.colorbar(sc, ax=ax, label=xyz_labels[2])

    pc = plot_configuration or PlotConfiguration()
    addCMSBits(
        ax,
        [x.metadata for x in signal] + [background.metadata],
        extra_text=f"{common_metadata['pipeline']}",
        plot_configuration=pc,
    )
    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[1])

    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


@define
class Significance2D(BasePostprocessor):
    output_name: str
    group_xy_patterns: tuple[list[str], list[str]]
    xyz_labels: tuple[str, str, str]
    style: Style = field(factory=Style)
    significance_type: SignificanceType = SignificanceType.poisson_basic
    significance_calculation: SignificanceCalculation = (
        SignificanceCalculation.single_bin
    )
    scale: Literal["log", "linear"] = "linear"

    def getRunFuncs(self, group, prefix=None):
        background = group["background"]
        signal = group["signal"]
        common_meta = commonDict(it.chain(background, signal))
        output_path = doFormatting(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)

        yield ft.partial(
            makeSignificance2D,
            signal=signal,
            background=background[0],
            common_metadata=common_meta,
            output_path=output_path,
            significance_type=self.significance_type,
            significance_calculation=self.significance_calculation,
            xy_pattern=self.group_xy_patterns,
            xyz_labels=self.xyz_labels,
            plot_configuration=pc,
            style=self.style,
        )
