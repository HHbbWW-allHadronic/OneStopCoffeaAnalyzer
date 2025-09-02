import pickle as pkl
from pathlib import Path



def exportHist(
    packaged_hist,
    output_path,
):
    ret = {}
    p = packaged_hist.sector_parameters
    k = (
        p.dataset.name,
        p.region_name,
    )
    h = packaged_hist.histogram
    ret[k] = {
        "params": p.model_dump(),
        "hist": h,
    }
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as f:
        pkl.dump(ret, f)
