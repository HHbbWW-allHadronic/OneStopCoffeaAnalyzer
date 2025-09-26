import pickle as pkl
from pathlib import Path
import lz4.frame


def exportHist(
    packaged_hist,
    output_path,
    overwrite=False,
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
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving {output_path}")
    if output_path.exists() and not overwrite:
        return
    with lz4.frame.open(output_path, "wb") as f:
        pkl.dump(ret, f)
