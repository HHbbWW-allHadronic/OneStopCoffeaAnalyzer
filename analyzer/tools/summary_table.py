import functools as ft
import operator as op

from analyzer.datasets import DatasetRepo, EraRepo
from analyzer.utils.querying import dictMatches
from rich import print


def texEscape(s):
    return s.replace("_", "\\_")


def makeTableFromDict(data):
    keys = list(data[0].keys())
    header = keys
    ret = [keys] + sorted([[d[k] for k in keys] for d in data], key=op.itemgetter(0))
    return ret


def texTable(elems, separator=" & ", force_max=None, col_format_funcs=None):
    col_format_funcs = col_format_funcs or {}
    elems = [
        [
            texEscape((col_format_funcs.get(i, str) if j > 0 else str)(x))
            for i, x in enumerate(y)
        ]
        for j, y in enumerate(elems)
    ]
    max_lens = [force_max or max(len(x) for x in y) for y in zip(*elems)]
    row_format = separator.join(f"{{: <{l}}}" for l in max_lens)
    res = " \\\\\n".join(row_format.format(*e) for e in elems)
    return res


def createEraTable(era_repo, query):
    ret = []
    for era_name in era_repo:
        era = era_repo[era_name]
        era_dict = era.model_dump()
        matches = dictMatches(query, era_dict)
        ret.append(matches)
    return ret
