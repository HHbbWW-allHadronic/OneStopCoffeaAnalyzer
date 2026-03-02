from __future__ import annotations

import functools as ft
import itertools as it
import copy
from analyzer.utils.structure_tools import (
    dotFormat,
    dictToDot,
    commonDict,
    ItemWithMeta,
)
from .processors import BasePostprocessor
from .style import StyleSet
from .plots.common import PlotConfiguration
from attrs import define, field
import pickle as pkl
from pathlib import Path
import lz4.frame
import numpy as np
from typing import Literal

import hist
import gzip
import correctionlib.convert
import correctionlib.schemav2 as cs


def exportEfficiencyToCorrectionLib(
    num,
    den,
    common_meta,
    ratio_type,
    output_path,
    correction_name,
    version=1,
    description="Ratio correction",
    diagnostic_plots=False,
    diagnostic_output_path=None,
    plot_configuration=None,
    style_set=None,
):
    from .plots.plots_1d import getRatioAndUnc

    num_hist, num_meta = num.item.histogram, num.metadata
    den_hist, den_meta = den.item.histogram, den.metadata

    if num_hist.axes != den_hist.axes:
        raise ValueError("Numerator and denominator axes do not match")

    n_vals = num_hist.values()
    d_vals = den_hist.values()

    ratio, unc = getRatioAndUnc(n_vals, d_vals, uncertainty_type=ratio_type)

    axes = list(num_hist.axes)
    # sys_ax = hist.axis.StrCategory(["nominal", "up", "down"], name="systematic")
    # axes = [sys_ax] + axes
    ratio = np.nan_to_num(ratio, nan=1.0)
    unc_0 = np.nan_to_num(unc[0], nan=0.0)
    unc_1 = np.nan_to_num(unc[1], nan=0.0)

    all_corrs = []
    for name, val in [
        ("nominal", ratio),
        ("down", ratio - unc_0),
        ("up", ratio + unc_1),
    ]:
        ret_hist = hist.Hist(*axes, storage=hist.storage.Double())
        ret_hist[...] = val
        ret_hist.name = f"{correction_name}_{name}"
        ret_hist.label = "output"
        corr = correctionlib.convert.from_histogram(ret_hist)
        corr.description = description
        corr.data.flow = "clamp"
        all_corrs.append(corr)

    cset = cs.CorrectionSet(schema_version=2, corrections=all_corrs)

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with gzip.open(output_path, "wt") as fout:
        fout.write(cset.model_dump_json(exclude_unset=True))

    if diagnostic_plots and diagnostic_output_path is not None:
        generateDiagnosticPlots(
            num,
            den,
            common_meta,
            ratio_type,
            diagnostic_output_path,
            corr=corr,
            plot_configuration=plot_configuration,
            style_set=style_set,
            ratio_unc=(ratio,*unc),
        )

def generateDiagnosticPlots(
    num,
        *args,**kwargs
):
    num_hist, num_meta = num.item.histogram, num.metadata
    generateDiagnosticPlots1D(num,*args,**kwargs)

    if len(num_hist.axes) == 2:
        generateDiagnosticPlots2D(num,*args,**kwargs)

        

def generateDiagnosticPlots1D(
    num,
    den,
    common_meta,
    ratio_type,
    base_output_path,
    corr=None,
    plot_configuration=None,
    style_set=None,
        ratio_unc=None,
):
    import matplotlib.pyplot as plt
    from .plots.utils import saveFig
    from .plots.annotations import labelAxis, addCMSBits
    from .plots.common import PlotConfiguration
    from analyzer.core.results import Histogram
    from .style import Styler
    from .plots.plots_1d import plotRatio
    import numpy as np

    num_hist, num_meta = num.item.histogram, num.metadata
    den_hist, den_meta = den.item.histogram, den.metadata

    pc = plot_configuration or PlotConfiguration()
    base_output_path = str(base_output_path)
    ext = pc.image_type
    if not base_output_path.endswith(f".{ext}"):
        ext_dot = "." + base_output_path.split(".")[-1]
    else:
        ext_dot = f".{ext}"

    for i, ax in enumerate(num_hist.axes):
        n_proj = num_hist.project(ax.name)
        d_proj = den_hist.project(ax.name)
        n_to_pass = ItemWithMeta(
            Histogram(name=num.item.name, histogram=n_proj, axes=[]), num.metadata
        )
        d_to_pass = ItemWithMeta(
            Histogram(name=den.item.name, histogram=d_proj, axes=[]), den.metadata
        )
        proj_output_path = base_output_path.replace(
            ext_dot, f"_proj_{ax.name}{ext_dot}"
        )

        plotRatio(
            [d_to_pass],
            [n_to_pass],
            proj_output_path,
            style_set,
            ratio_ylim=(0.0, 1.1),
            no_stack=True,
            ratio_type="efficiency",
            ratio_height=0.4,
            scale="log",
            plot_configuration=plot_configuration,
        )

def generateDiagnosticPlots2D(
    num,
    den,
    common_meta,
    ratio_type,
    base_output_path,
    corr=None,
    plot_configuration=None,
    style_set=None,
        ratio_unc=None,
):
    import matplotlib.pyplot as plt
    from .plots.utils import saveFig
    from .plots.annotations import labelAxis, addCMSBits
    from .plots.common import PlotConfiguration
    from analyzer.core.results import Histogram
    from .style import Styler
    from .plots.plots_1d import plotRatio
    import numpy as np


    pc = plot_configuration or PlotConfiguration()
    base_output_path = str(base_output_path)
    ext = pc.image_type
    if not base_output_path.endswith(f".{ext}"):
        ext_dot = "." + base_output_path.split(".")[-1]
    else:
        ext_dot = f".{ext}"


    ratio = num.item.histogram.copy(deep=True)
    up = num.item.histogram.copy(deep=True)
    down = num.item.histogram.copy(deep=True)
    ratio[...] = ratio_unc[0]
    up[...] = ratio_unc[1]
    down[...] = ratio_unc[2]

    for h,name in [(ratio,"eff"), (up,"up"), (down, "down")]:
        styler = Styler(style_set)
        fig, ax = plt.subplots(layout="constrained")
        h.plot2d(ax=ax)
        labelAxis(ax, "y", h.axes)
        labelAxis(ax, "x", h.axes)

        addCMSBits(
            ax,
            [common_meta],
            extra_text=f"{common_meta['pipeline']}",
            text_color="black",
            plot_configuration=pc,
        )
        this_output_path = base_output_path.replace(
            ext_dot, f"2d_{name}{ext_dot}"
        )
        saveFig(fig, this_output_path, extension=pc.image_type)
        plt.close(fig)



@define
class CorrectionLibEff(BasePostprocessor):
    output_name: str
    correction_name: str
    ratio_type: Literal["poisson", "efficiency"] = "efficiency"
    diagnostic_plots: bool = False
    diagnostic_output_name: str | None = None
    style_set: str | StyleSet = field(factory=StyleSet)

    def getRunFuncs(self, group, prefix=None):
        numerator = group["numerator"]
        denominator = group["denominator"]

        if len(numerator) != 1 or len(denominator) != 1:
            raise RuntimeError(
                "CorrectionLibExport expects exactly 1 numerator and 1 denominator."
            )

        num = numerator[0]
        den = denominator[0]
        common_meta = commonDict([num, den])
        output_path = dotFormat(
            self.output_name, prefix=prefix, **dict(dictToDot(common_meta))
        )

        diag_output_path = None
        pc = None
        if self.diagnostic_plots and self.diagnostic_output_name is not None:
            diag_output_path = dotFormat(
                self.diagnostic_output_name,
                prefix=prefix,
                **dict(dictToDot(common_meta)),
            )
            pc = self.plot_configuration.makeFormatted(common_meta)

        yield ft.partial(
            exportEfficiencyToCorrectionLib,
            num,
            den,
            common_meta,
            self.ratio_type,
            output_path,
            self.correction_name,
            diagnostic_plots=self.diagnostic_plots,
            diagnostic_output_path=diag_output_path,
            plot_configuration=pc,
            style_set=self.style_set,
        )
