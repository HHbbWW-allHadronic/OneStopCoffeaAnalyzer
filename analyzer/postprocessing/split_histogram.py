import copy
import enum
import itertools as it
from collections import OrderedDict



class Mode(str, enum.Enum):
    Split = "split"
    Sum = "sum"


def splitHistogram(histogram, axis_options, allow_missing=False):
    h = copy.deepcopy(histogram)
    if allow_missing:
        axis_options = {x: y for x, y in axis_options if x in histogram.axes}
    sum_axes_names = [a for a, y in axis_options.items() if y == Mode.Sum]
    split_axes = [histogram.axes[y] for a, y in axis_options.items() if y == Mode.Split]
    val_axes = {a: y for a, y in axis_options.items() if not isinstance(y, Mode)}
    h = h[{x: sum for x in sum_axes_names}]
    if not split_axes:
        return h
    possible_values = {**{x.name: list(x) for x in split_axes}, **val_axes}
    options = OrderedDict(possible_values)
    ret = {x: h[dict(zip(options.keys(), x))] for x in it.product(*options.values())}
    return ret
