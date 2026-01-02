from collections import defaultdict

TRANSFORM_REGISTRY = defaultdict(dict)


def addToRegistry(result_type):
    def inner(func):
        TRANSFORM_REGISTRY[result_type][func.name] = func
        return func

    return inner
