
import functools as ft
import itertools as it
import string



def getNested(d, s):
    parts = s.split(".")

    def getK(di, p):
        return di[p]

    ret = ft.reduce(getK, parts, d)
    return ret


def doFormatting(s, sector_params, **kwargs):
    parsed = string.Formatter().parse(s)
    d = sector_params.model_dump()
    s = ""
    for x in parsed:
        s += x[0]
        if x[1] is not None:
            if x[1] in kwargs:
                s += kwargs[x[1]]
            else:
                s += getNested(d, x[1])
    return s


def groupBy(fields, data):
    def k(v):
        return tuple([getNested(v.sector_params.model_dump(), x) for x in fields])

    grouped = it.groupby(sorted(data, key=k), k)
    ret = [(dict(zip(fields, x)), list(y)) for x, y in grouped]
    return ret
