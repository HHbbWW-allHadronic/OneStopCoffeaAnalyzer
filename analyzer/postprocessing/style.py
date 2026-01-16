import logging

import copy
import matplotlib as mpl
from analyzer.utils.querying import BasePattern
import matplotlib.typing as mplt
from cycler import cycler
from attrs import define, field, asdict, filters
from pathlib import Path
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import mplhep
from analyzer.configuration import CONFIG
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

logger = logging.getLogger("analyzer")


cms_colors = {
    # blues
    "cms-blue": "#5790fc",
    "cms-blue-dark": "#3f90da",
    "cms-cyan": "#92dadd",
    # oranges / yellows
    "cms-orange": "#f89c20",
    "cms-orange-light": "#ffa90e",
    "cms-orange-dark": "#e76300",
    "cms-gold": "#b9ac70",
    # reds
    "cms-red": "#e42536",
    "cms-red-dark": "#bd1f01",
    # purples
    "cms-purple": "#964a8b",
    "cms-purple-dark": "#7a21dd",
    "cms-violet": "#832db6",
    # neutrals
    "cms-gray": "#9c9ca1",
    "cms-gray-dark": "#717581",
    "cms-gray-cool": "#94a4a2",
    # earth / accent
    "cms-brown": "#a96b59",
}

cms_colors_6 = [
    cms_colors["cms-blue"],
    cms_colors["cms-orange"],
    cms_colors["cms-red"],
    cms_colors["cms-purple"],
    cms_colors["cms-gray"],
    cms_colors["cms-purple-dark"],
]

cms_colors_10 = [
    cms_colors["cms-blue-dark"],
    cms_colors["cms-orange-light"],
    cms_colors["cms-red-dark"],
    cms_colors["cms-gray-cool"],
    cms_colors["cms-violet"],
    cms_colors["cms-brown"],
    cms_colors["cms-orange-dark"],
    cms_colors["cms-gold"],
    cms_colors["cms-gray-dark"],
    cms_colors["cms-cyan"],
]


def loadStyles():
    font_dir = str(Path(CONFIG.post.static_resource_path) / "fonts")
    mplhep.style.use("CMS")
    # str(Path(CONFIG.STYLE_PATH) / "style.mplstyle")
    mpl.rcParams["figure.constrained_layout.use"] = True
    font_files = font_manager.findSystemFonts(fontpaths=[font_dir])
    for font in font_files:
        font_manager.fontManager.addfont(font)
    mcolors.get_named_colors_mapping().update(cms_colors)
    default_cycler = cycler(linestyle=["-", "--", ":", "-."]) * cycler(
        color=cms_colors_10
    )

    plt.rcParams["axes.prop_cycle"] = default_cycler


@define
class Style:
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
            step=("color",),
            fill=("color", "alpha"),
            band=("color",),
            errorbar=("color", "markersize", "marker"),
        )
        ret = asdict(self, filter=filters.include(*mapping[plottype]))
        ret.setdefault("linewidth", mpl.rcParams["lines.linewidth"])
        if include_type:
            ret["histtype"] = plottype
        if prepend:
            ret = {f"{prepend}_{x}": y for x, y in ret.items()}
        return ret


@define
class StyleRule:
    style: Style
    pattern: BasePattern | None = None


@define
class StyleSet:
    styles: list[StyleRule] = field(factory=list)

    def getStyle(self, sector_params, other_data=None):
        for style_rule in self.styles:
            if style_rule.pattern is None:
                return style_rule.style
            elif style_rule.pattern.match(sector_params):
                return style_rule.style
        return None


class Styler:
    def __init__(self, style_set, expected_num=6):
        self.style_set = style_set
        self.expected_num = expected_num
        self.cycler = plt.rcParams["axes.prop_cycle"]
        self.cycle_iter = iter(self.cycler)

    def getStyle(self, sector_params):
        found = self.style_set.getStyle(sector_params)
        if found is not None:
            if found.color is None:
                found = copy.deepcopy(found)
                c = next(self.cycle_iter)
                found.color = c["color"]
            return found

        c = next(self.cycle_iter)
        color, linestyle = c["color"], c["linestyle"]
        return Style(color=color, linestyle=linestyle)
