import json
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting.core_plots import *
import argparse
import pandas as pd
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.collections import RegularPolyCollection, PatchCollection
from pathlib import Path
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union, Callable
import operator as op
from analyzer.plotting.core_plots import *




def parseArguments():
    parser = argparse.ArgumentParser(description="Generate fit scans")
    parser.add_argument("-i", "--input", type=str, help="Input file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    parser.add_argument("-f", "--filter", nargs="*", type=str, help="Output file path")
    parser.add_argument("-x","--x-field", type=str)
    parser.add_argument("-y","--y-field", type=str)
    parser.add_argument("-w","--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("-v", type=str)
    parser.add_argument("--text", type=str)
    return parser.parse_args()


operators = {"=": op.eq, "<": op.lt, ">": op.gt, "<=": op.le, ">=": op.ge}


@dataclass
class Filter:
    field: str
    value: str| int| float
    operator: Callable[[str| int| float, str| int| float], bool]

    @staticmethod
    def make(string):
        for x, o in operators.items():
            if x in string:
                field, value = string.split(x)
                operator = o
        return Filter(field, json.loads(value), operator)

    def __call__(self, data):
        return self.operator(data[self.field], self.value)


def main():
    loadStyles()
    args = parseArguments()
    filters = [Filter.make(s) for s in args.filter]
    print(filters)

    data = json.load(open(args.input, "r"))
    for f in filters:
        data = [x for x in data if f(x)]
    print(len(data))
    rects = [
        AnnotRectangle(
            x=d["inject_signal_rate"],
            y=d["inject_signal_params"][1],
            w=1,
            h=10,
            text=f"{round(d['reduced_chi2'],2)}, {round(d['length_scale'])}",
            value=d["reduced_chi2"],
        )
        for d in data
    ]
    fig, ax = plt.subplots()
    outp = Path(args.output)
    drawRectangles(ax, rects)
    ax.set_xlabel("Signal Rate")
    ax.set_ylabel(r"Injected Gaussian $\sigma$")
    ax.cax.set_ylabel(r"Reduced $\chi^2$")
    fig.savefig(outp)


if __name__ == "__main__":
    main()
