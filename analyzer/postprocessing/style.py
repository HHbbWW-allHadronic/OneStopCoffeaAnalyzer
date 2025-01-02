import logging

import matplotlib.typing as mplt
import matplotlib as mpl
from analyzer.core.specifiers import SectorSpec
from cycler import cycler
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Style(BaseModel):
    color: mplt.ColorType | None = None
    linestyle: mplt.LineStyleType | None = None
    drawstyle: mplt.DrawStyleType | None = None
    fillstyle: mplt.FillStyleType | None = None
    capstyle: mplt.CapStyleType | None = None
    joinstyle: mplt.JoinStyleType | None = None
    fill: bool | None = None
    fill_opacity: float | None = None
    fill_hatching: str | None = None
    line_width: float | None = None
    markersize: float | None = 5
    marker: str | None = "o"

    def get(self, type, prepend=None):
        mapping = dict(
            step=("color",),
            fill=("color",),
            band=("color",),
            errorbar=("color", "markersize", "marker"),
        )
        ret = self.model_dump(include=mapping[type])
        ret.setdefault("linewidth", mpl.rcParams["lines.linewidth"])
        if prepend:
            ret = {f"{prepend}_{x}": y for x, y in ret.items()}
        return ret


class StyleRule(BaseModel):
    sector_spec: SectorSpec
    style: Style


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

    def getStyle(self, sector_params):
        for style_rule in self.styles:
            if style_rule.sector_spec.passes(sector_params):
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
