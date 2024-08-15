import awkward as ak

def isMC(sample_info):
    return sample_info.sample_type == "MC"

def isData(sample_info):
    return sample_info.sample_type == "Data"

def numMatching(a1, a2):
    c = ak.cartesian([a1, a2])
    x, y = ak.unzip(c)
    m = x == y
    return ak.sum(m, axis=1)
