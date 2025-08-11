import pydantic as pyd
import matplotlib.typing as mplt
import copy
from typing import Optional
from ..grouping import doFormatting


class PlotConfiguration(pyd.BaseModel):
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


    def makeFormatted(self, params):
        ret = copy.deepcopy(self)
        if ret.extra_text:
            ret.extra_text = doFormatting(ret.extra_text, params)
        if ret.cms_text:
            ret.cms_text = doFormatting(ret.cms_text, params)
        return ret


