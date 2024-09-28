import contextlib

import numpy as np
import itertools as it

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import enum
import hist


class _Split(object):
    def __new__(cls):
        return NoParam

    def __reduce__(self):
        return (_NoParamType, ())


Split = object.__new__(_Split)


def reformatHist(
    hist,
    axis_opts,
):
    h = hist
    axes_names = {x.name for x in h.axes}
    for n in all_axis_opts.keys():
        if n not in axes_names:
            raise KeyError(f"Name {n} is not an axis in {h}")
    to_split = [x for x, y in all_axis_opts.items() if y is Split]
    all_axis_opts = {x: y for x, y in all_axis_opts.items() if y is not Split}
    h = h[all_axis_opts]
    if to_split:
        split_axes = [list(x) for x in h.axes if x.name in to_split]
        split_names = [x.name for x in h.axes if x.name in to_split]
        for combo in it.product(*split_axes):
            f = dict(zip(split_names, (hist.loc(x) for x in combo)))
            yield h[f]
    else:
        yield h


def getNormalized(hist, dataset_axis=None, sample_manager=None):
    ret = hist.copy(deep=True)
    if dataset_axis is None:
        val, var = ret.values(flow=True), ret.variances(flow=True)
        s = ret.sum(flow=True).value
        ret[...] = np.stack([val / s, var / s**2], axis=-1)
    else:
        for i,x in enumerate(ret.axes[dataset_axis]):
            this = ret[{dataset_axis: x}]
            val, var = this.values(flow=True), this.variances(flow=True)
            s = this.sum(flow=True).value
            ret[{dataset_axis: x}] = np.stack([val / s, var / s**2], axis=-1)
            if sample_manager is not None:
                sample_manager.weights_normalized[i] *= 1/s
    return ret


def addAxesToHist(
    ax,
    num_top=0,
    num_bottom=0,
    num_left=0,
    num_right=0,
    top_pad=0.2,
    right_pad=0.2,
    left_pad=0.2,
    bottom_pad=0.2,
    **plotopts,
):
    divider = make_axes_locatable(ax)
    ax.top_axes = []
    ax.bottom_axes = []
    ax.right_axes = []
    ax.left_axes = []
    for i in range(num_top):
        ax_histx = divider.append_axes("top", 1, pad=top_pad, sharex=ax)
        ax.top_axes.append(ax_histx)

    for i in range(num_bottom):
        ax_histx = divider.append_axes("bottom", 1, pad=bottom_pad, sharex=ax)
        ax.bottom_axes.append(ax_histx)

    for i in range(num_left):
        ax_histx = divider.append_axes("left", 1, pad=left_pad, sharey=ax)
        ax.left_axes.append(ax_histx)

    for i in range(num_right):
        ax_histx = divider.append_axes("right", 1, pad=right_pad, sharey=ax)
        ax.right_axes.append(ax_histx)
    return ax


@contextlib.contextmanager
def subplots_context(*args, close_after=True, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    yield fig, ax
    if close_after:
        plt.close(fig)


class AxisOptions(enum.Enum):
    Split = enum.auto()
    Sum = enum.auto()


def splitHistogram(h, axes_options, split_separator="__", split_val_separator="eq"):
    a = set(a.name for a in h.axes)
    b = set(axes_options)
    if not b <= a:
        raise ValueError()
    sum_axes = {x: sum for x, y in axes_options.items() if y == AxisOptions.Sum}
    split_names = [
        x.name for x in h.axes if axes_options.get(x.name, None) == AxisOptions.Split
    ]
    split_values = [list(x) for x in h.axes if x.name in split_names]
    hsummed = h[sum_axes]
    ret = {}
    for combo in it.product(*split_values):
        vals = dict(zip(split_names, combo))
        cut = {x: hist.loc(y) for x, y in vals.items()}
        updated = hsummed[cut]
        update_name = split_separator.join(
            f"{x}{split_val_separator}{y}" for x, y in vals.items()
        )
        ret[update_name] = updated
    return ret


def splitHistDict(hname, hist_dict, axes_options, split_val_separator="__", **kwargs):
    split = {
        x: splitHistogram(y, axes_options, split_val_separator, **kwargs)
        for x, y in hist_dict.items()
    }
    keys = set(it.chain.from_iterable(x.keys() for x in split.values()))
    s = split_val_separator if len(keys) > 1 else ""

    r = {
        hname
        + (split_val_separator if key else "")
        + key: {k: split[k][key] for k in split}
        for key in keys
    }
    return r
