import itertools as it
import logging
from collections import namedtuple
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib

import pydantic as pyd
from analyzer.core.specifiers import SampleSpec, SectorSpec
from rich import print

from .split_histogram import Mode
from .style import Style

import mplhep


logger = logging.getLogger(__name__)


StyleLike = Union[Style, str]




def fillString(string, params, **rest):
    return string.format(**rest, **params)


class BasePostprocessor(pyd.BaseModel):
    output_dir: str
    styles: list[Union[Style, str]]

    def resolveStyle(self, dataset, region):
        pass


def groupBy(fields, data):
    Group = namedtuple("Group", fields)

    def k(v):
        return Group(*[v.sector_params[x] for x in fields])

    grouped = it.groupby(sorted(data,key=k), k)
    ret = {x: list(y) for x, y in grouped}
    return ret


class Histogram1D(pyd.BaseModel):

    histogram_names: list[str]
    to_plot: SectorSpec

    groupby: list[str] = ["era", "region_name"]
    output_name: str = "{histogram_name}{categories}"

    axis_options: Optional[dict[str, Union[Mode, str, int]]] = None
    normalize: bool = False
    lumi_str: Optional[str] = None
    era_str: Optional[str] = None
    energy_str: Optional[str] = None
    extra_text: Optional[str] = None
    cms_text: Optional[str] = None

    x_scale: Optional[str] = None
    y_scale: Optional[str] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = groupBy(self.groupby, sectors)
        ret = []
        for histogram in self.histogram_names:
            for group, sectors in r.items():
                fig,ax=plt.subplots()
                for sector in sectors:
                    print(sector)
                    h = sector.histograms[histogram].get()
                    mplhep.histplot(h,binwnorm=1, ax=ax,label=sector.sector_params.title)
                ax.set_xlabel("HT [GeV]")
                ax.legend()
                t ="_".join(group)
                fig.tight_layout()
                fig.savefig(f"plots/{histogram}_{t}.pdf")

            



class RatioPlot(BasePostprocessor):
    name_format: str
    numerator: list[SectorSpec]
    denominator: list[SectorSpec]

    def __init__(self, default_configuration):
        pass

    def getExe(self, dataset_results):
        pass


if __name__ == "__main__":
    from analyzer.core import AnalysisResult
    matplotlib.use("Agg")
    #loadStyles()
    mplhep.style.use("CMS")


    result = AnalysisResult.fromFile("test.pkl")
    result = result.getResults()
    h = Histogram1D(
        histogram_names=["HT"],
        to_plot=SectorSpec(sample_spec=SampleSpec(name="signal*")),
    )
    r = h.getExe(result)
    print(r)
        

