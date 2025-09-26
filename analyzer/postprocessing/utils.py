from analyzer.core.results import MultiSampleResult
from rich import print
from analyzer.utils.structure_tools import dictToFrozen
from collections import defaultdict


def gatherByPattern(params_map, pattern, limit_capture_fields=None):
    params_dict = defaultdict(set)
    for params, path in params_map:
        if not pattern.match(params, strict=False):
            continue
        p = pattern.capture(params)
        if limit_capture_fields and isinstance(p, dict):
            p = {x: y for x, y in p.items() if x in limit_capture_fields}

        p = dictToFrozen(p)
        params_dict[p].add(path)

    return set(frozenset(sorted(x)) for x in params_dict.values())
