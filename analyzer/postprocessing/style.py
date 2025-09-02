import logging

import matplotlib as mpl
import matplotlib.typing as mplt
from analyzer.utils.querying import NestedPatternExpression
from cycler import cycler
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class Style(BaseModel):
    color: mplt.ColorType | None = None
    plottype: str = "step"
    linestyle: mplt.LineStyleType | None = None
    drawstyle: mplt.DrawStyleType | None = None
    fillstyle: mplt.FillStyleType | None = None
    capstyle: mplt.CapStyleType | None = None
    joinstyle: mplt.JoinStyleType | None = None
    fill: bool | None = None
    alpha: float | None = None
    fill_hatching: str | None = None
    line_width: float | None = None
    markersize: float | None = 5
    marker: str | None = "o"
    yerr: bool = True
    y_min: float | None = None
    legend: bool = True
    legend_font: int | None = None

    def get(self, plottype=None, prepend=None, include_type=True):
        if plottype is None:
            plottype = self.plottype
        mapping = dict(
            step=( "color",),
            fill=( "color", "alpha"),
            band=( "color",),
            errorbar=( "color", "markersize", "marker"),
        )
        ret = self.model_dump(include=mapping[plottype])
        ret.setdefault("linewidth", mpl.rcParams["lines.linewidth"])
        if include_type:
            ret["histtype"] = plottype
        if prepend:
            ret = {f"{prepend}_{x}": y for x, y in ret.items()}
        return ret


class StyleRule(BaseModel):
    style: Style
    sector_spec: NestedPatternExpression | None = None


cms_colors_6 = [
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
]

cms_colors_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]


class StyleSet(BaseModel):
    styles: list[StyleRule] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def flattenStyle(cls, data):
        if isinstance(data, list):
            return {"styles": data}
        return data

    def getStyle(self, sector_params, other_data=None):
        for style_rule in self.styles:
            if style_rule.sector_spec is None:
                return style_rule.style
            elif style_rule.sector_spec.match(sector_params):
                logger.debug(
                    f"Found matching style rule for {sector_params.dataset.name}"
                )
                return style_rule.style
        logger.debug(
            f"Did not find matching style rule for {sector_params.dataset.name}"
        )
        return None


class Styler:
    def __init__(self, style_set, expected_num=6):
        self.style_set = style_set
        self.expected_num = expected_num
        self.cycler = cycler(linestyle=["-", "--", ":", "-."]) * cycler(
            color=cms_colors_10
        )
        self.cycle_iter = iter(self.cycler)

    def getStyle(self, sector_params):
        found = self.style_set.getStyle(sector_params)
        if found is not None:
            return found

        c = next(self.cycle_iter)
        color, linestyle = c["color"], c["linestyle"]
        return Style(color=color, linestyle=linestyle)
