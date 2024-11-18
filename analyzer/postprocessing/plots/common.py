import pydantic as pyd
from typing import Optional

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





