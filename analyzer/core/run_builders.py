from collections import defaultdict


def toTuples(d):
    return {(x, y): v for x, s in d.items() for y, v in s.items()}


def fromTuples(d):
    ret = defaultdict(dict)
    for (k1, k2), v in d.items():
        ret[k1][k2] = v
    return dict(ret)


def buildCombos(spec, tag):
    ret = []
    tup = toTuples(spec.getTags(tag))
    central = {k: v.default_value for k, v in tup.items()}
    for k, v in tup.items():
        for p in v.possible_values:
            if p == v.default_value:
                continue
            c = copy.deepcopy(central)
            c[k] = p
            ret.append([p, c])

    ret = [[n, fromTuples(x)] for n, x in ret]

    return ret


def buildVariations(spec, metadata=None):
    weights = buildCombos(spec, "weight_variation")
    shapes = buildCombos(spec, "shape_variation")
    all_vars = [["central", {}]] + weights + shapes
    return all_vars
