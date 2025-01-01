import pickle as pkl
from pathlib import Path


from ..grouping import doFormatting


def exportHist(
    histogram_name,
    group_parameters,
    sectors,
    output_name,
):
    ret = {}
    for sector in sectors:
        p = sector.sector_params
        k = (
            p.dataset.name,
            p.region_name,
        )
        ret[k] = {
            "params": sector.sector_params.model_dump(),
            "hist_name": histogram_name,
            "hist_collection": sector.histograms[histogram_name].model_dump(),
        }
    o = doFormatting(output_name, group_parameters, histogram_name=histogram_name)
    Path(o).parent.mkdir(exist_ok=True, parents=True)
    with open(o, "wb") as f:
        pkl.dump(ret, f)
