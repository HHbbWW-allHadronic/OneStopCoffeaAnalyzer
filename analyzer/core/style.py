from typing import Optional

import matplotlib.typing as mplt
from analyzer.core.specifiers import SampleSpec, SectorSpec
from pydantic import BaseModel


class Style(BaseModel):
    color: Optional[mplt.ColorType] = None
    line_stype: Optional[mplt.LineStyleType] = None
    draw_style: Optional[mplt.DrawStyleType] = None
    fill_style: Optional[mplt.FillStyleType] = None
    cap_style: Optional[mplt.CapStyleType] = None
    join_stye: Optional[mplt.JoinStyleType] = None


class StyleRule(BaseModel):
    sector_spec: SectorSpec
    style: Style




