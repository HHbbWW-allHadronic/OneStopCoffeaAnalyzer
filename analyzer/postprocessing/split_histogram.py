import copy
import itertools as it
import hist
import enum
import itertools as it
from collections import OrderedDict


class Mode(str, enum.Enum):
    Split = "Split"
    Sum = "Sum"
    Or = "Or"
    Rebin2 = "Rebin2"
    Rebin3 = "Rebin3"
    Rebin4 = "Rebin4"


def splitHistogram(
    histogram, axis_options=None, allow_missing=False, return_labels=False
):
    if "REST" in axis_options:
        other_axes = [x.name for x in histogram.axes if x.name not in axis_options]
        axis_options = {x: axis_options["REST"] for x in other_axes} | {
            x: y for x, y in axis_options.items() if x != "REST"
        }

    h = copy.deepcopy(histogram)
    if not axis_options:
        if return_labels:
            return h, []
        else:
            return h
    if allow_missing:
        axis_options = {x: y for x, y in axis_options if x in histogram.axes}

    or_axes = [h.axes[a].name for a, y in axis_options.items() if y == Mode.Or]

    for ora in or_axes:
        if isinstance(h.axes[ora], hist.axis.Boolean):
            raise RuntimeError(f"Can only use Or on boolean axes")
    new_h = None
    if or_axes:
        for x in it.product(*[[0, 1]] * len(or_axes)):
            if not any(x):
                continue
            if new_h is None:
                new_h = h[dict(zip(or_axes, x))]
            else:
                new_h += h[dict(zip(or_axes, x))]
        h = new_h

    sum_axes_names = [a for a, y in axis_options.items() if y == Mode.Sum]
    val_axes = {a: y for a, y in axis_options.items() if not isinstance(y, Mode)}
    rebin2_axes_names = [a for a, y in axis_options.items() if y == Mode.Rebin2]
    rebin3_axes_names = [a for a, y in axis_options.items() if y == Mode.Rebin3]
    rebin4_axes_names = [a for a, y in axis_options.items() if y == Mode.Rebin4]

    first = {
        **val_axes,
        **{x: sum for x in sum_axes_names},
        **{x: hist.rebin(2) for x in rebin2_axes_names},
        **{x: hist.rebin(3) for x in rebin3_axes_names},
        **{x: hist.rebin(4) for x in rebin4_axes_names},
    }
    h = h[first]

    split_axes = [h.axes[a] for a, y in axis_options.items() if y == Mode.Split]

    if not split_axes:
        if return_labels:
            return h, []
        else:
            return h
    possible_values = {(x.name): list(x) for x in split_axes}
    options = OrderedDict(possible_values)
    ret = {
        x: h[dict(zip(options.keys(), map(hist.loc, x)))]
        for x in it.product(*options.values())
    }
    if return_labels:
        return ret, [(x.name or x.label) for x in split_axes]
    else:
        return ret
