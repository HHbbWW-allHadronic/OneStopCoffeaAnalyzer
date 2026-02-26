from __future__ import annotations

import functools as ft
from typing import Literal
import itertools as it
from .style import StyleSet
from analyzer.utils.structure_tools import (
    ItemWithMeta,
    commonDict,
    dictToDot,
    dotFormat,
)
from .processors import BasePostprocessor
from .plots.plots_1d import plotOne, plotRatio, plotRatioOfRatios
from .plots.plots_2d import plot2D
from attrs import define, field
from rich import print

ResultSet = list[list[ItemWithMeta]]


@define
class Histogram1D(BasePostprocessor):
    output_name: str
    style_set: str | StyleSet = field(factory=StyleSet)
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False

    def getRunFuncs(self, group, prefix=None):
        if isinstance(group, dict):
            unstacked = group["unstacked"]
            stacked = group["stacked"]
        else:
            unstacked = group
            stacked = None
        common_meta = commonDict(it.chain((stacked or []), (unstacked or [])))
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            plotOne,
            unstacked,
            stacked,
            common_meta,
            output_path,
            scale=self.scale,
            style_set=self.style_set,
            normalize=self.normalize,
            plot_configuration=pc,
        )


@define
class RatioPlot(BasePostprocessor):
    output_name: str
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    ratio_ylim: tuple[float, float] = (0, 2)
    ratio_hlines: list[float] = field(factory=lambda: [1.0])
    ratio_height: float = 0.5
    ratio_type: Literal["poisson", "poisson-ratio", "efficiency", "significance"] = (
        "poisson"
    )
    no_stack: bool = False

    def getRunFuncs(self, group, prefix=None):
        numerator = group["numerator"]
        denominator = group["denominator"]
        common_meta = commonDict(it.chain(numerator, denominator))
        output_path = dotFormat(
            self.output_name, prefix=prefix, **dict(dictToDot(common_meta))
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            plotRatio,
            denominator,
            numerator,
            output_path,
            self.style_set,
            normalize=self.normalize,
            ratio_ylim=self.ratio_ylim,
            ratio_type=self.ratio_type,
            scale=self.scale,
            ratio_hlines=self.ratio_hlines,
            ratio_height=self.ratio_height,
            no_stack=self.no_stack,
            plot_configuration=pc,
        )


@define
class RatioOfRatiosPlot(BasePostprocessor):
    output_name: str
    r1_label: str = "{numerator.dataset_title}/{denominator.dataset_title}"
    r2_label: str = "{numerator.dataset_title}/{denominator.dataset_title}"
    double_ratio_label: str = "Double Ratio"
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    ratio_ylim: tuple[float, float] = (0, 2)
    ratio_hlines: list[float] = field(factory=lambda: [1.0])
    ratio_height: float = 0.5
    ratio_type: Literal["poisson", "poisson-ratio", "efficiency", "significance"] = (
        "poisson"
    )

    def getRunFuncs(self, group, prefix=None):
        num_group = group["numerator"]
        den_group = group["denominator"]

        # Ensure we have exactly one item per component
        if len(num_group["numerator"]) != 1:
            raise ValueError(
                f"Expected exactly 1 item for num_group['numerator'], got {len(num_group['numerator'])}"
            )
        if len(num_group["denominator"]) != 1:
            raise ValueError(
                f"Expected exactly 1 item for num_group['denominator'], got {len(num_group['denominator'])}"
            )
        if len(den_group["numerator"]) != 1:
            raise ValueError(
                f"Expected exactly 1 item for den_group['numerator'], got {len(den_group['numerator'])}"
            )
        if len(den_group["denominator"]) != 1:
            raise ValueError(
                f"Expected exactly 1 item for den_group['denominator'], got {len(den_group['denominator'])}"
            )

        num_numerator = num_group["numerator"][0]
        num_denominator = num_group["denominator"][0]
        den_numerator = den_group["numerator"][0]
        den_denominator = den_group["denominator"][0]

        common_meta = commonDict(
            [num_numerator, num_denominator, den_numerator, den_denominator]
        )

        r1_meta = commonDict([num_numerator, num_denominator])
        r2_meta = commonDict([den_numerator, den_denominator])

        r1_label = dotFormat(self.r1_label, **dict(dictToDot(r1_meta)))
        r2_label = dotFormat(self.r2_label, **dict(dictToDot(r2_meta)))

        output_path = dotFormat(
            self.output_name, prefix=prefix, **dict(dictToDot(common_meta))
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            plotRatioOfRatios,
            num_numerator,
            num_denominator,
            den_numerator,
            den_denominator,
            output_path,
            self.style_set,
            r1_label=r1_label,
            r2_label=r2_label,
            double_ratio_label=self.double_ratio_label,
            normalize=self.normalize,
            ratio_ylim=self.ratio_ylim,
            ratio_type=self.ratio_type,
            scale=self.scale,
            ratio_hlines=self.ratio_hlines,
            ratio_height=self.ratio_height,
            plot_configuration=pc,
        )


@define
class Histogram2D(BasePostprocessor):
    output_name: str
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False

    def getRunFuncs(self, group, prefix=None):
        if len(group) != 1:
            raise RuntimeError()
        hist = group[0]
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, prefix=prefix, **dict(dictToDot(common_meta))
        )
        self.plot_configuration.makeFormatted(common_meta)
        yield ft.partial(
            plot2D,
            hist,
            common_meta,
            output_path,
            style_set=self.style_set,
            normalize=self.normalize,
            plot_configuration=self.plot_configuration,
            color_scale=self.scale,
        )
