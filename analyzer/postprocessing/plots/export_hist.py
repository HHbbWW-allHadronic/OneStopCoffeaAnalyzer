import pickle as pkl
from pathlib import Path


from ..utils import doFormatting


def exportHist(
    histogram_name,
    sectors,
    output_name,
):
    ret = {}
    for sector in sectors:
        p = sector.sector_params
        k = (
            p.dataset.name,
            p.region["region_name"],
        )
        ret[k] = {
            "params": sector.sector_params.model_dump(),
            "hist_name": histogram_name,
            "hist_collection": sector.histograms[histogram_name].model_dump(),
        }
    o = doFormatting(output_name, p, histogram_name=histogram_name)
    Path(o).parent.mkdir(exist_ok=True, parents=True)
    with open(o, "wb") as f:
        pkl.dump(ret, f)
