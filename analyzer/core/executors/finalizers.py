import dask
import awkward as ak

def basicFinalizer(data):
    if isinstance(data, list | dict):
        data = dask.compute(data)[0]
    elif isinstance(data, ak.Array):
        data = data.to_numpy()
    return data
