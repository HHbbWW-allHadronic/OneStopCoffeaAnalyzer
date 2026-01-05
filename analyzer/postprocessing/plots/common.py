import matplotlib.typing as mplt
import copy
from typing import Optional
from attrs import define


@define
class PlotConfiguration:
    lumi_text: Optional[str] = None
    extra_text: Optional[str] = None
    cms_text: Optional[str] = None
    cms_text_pos: int = 2
    cms_text_color: Optional[str] = None

    x_scale: Optional[str] = "linear"
    y_scale: Optional[str] = "linear"

    x_label: Optional[str] = None
    y_label: Optional[str] = None

    image_type: Optional[str] = None

    legend_fill_color: mplt.ColorType | None = None
    legend_fill_alpha: float | None = None

    def makeFormatted(self, meta):
        ret = copy.deepcopy(self)
        if ret.extra_text:
            ret.extra_text = ret.extra_text.format(**meta)
        if ret.cms_text:
            ret.cms_text = ret.cms_text.format(**meta)
        return ret
