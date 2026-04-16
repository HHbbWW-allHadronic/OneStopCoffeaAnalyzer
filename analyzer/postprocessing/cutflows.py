from __future__ import annotations

import functools as ft
from typing import Literal
from .style import StyleSet
from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    dotFormat,
)
from .processors import BasePostprocessor
from .plots.plots_1d import plotDictAsBars
from attrs import define, field


def _getCutflow(x):
    return getattr(x, "cutflow")

def _getOneCut(x):
    return getattr(x, "one_cut")

def _getNMinusOne(x):
    return getattr(x, "n_minus_one")


@define
class PlotSelectionFlow(BasePostprocessor):
    output_name: str
    style_set: str | StyleSet = field(factory=StyleSet)
    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    show_yields: bool = False

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        pc = self.plot_configuration.makeFormatted(common_meta)
        getters = {
                'Cutflow': _getCutflow,
                'OneCut': _getOneCut,
                'N-1': _getNMinusOne,
                }

        for ax_name, getter in getters.items():
            path = output_path.replace(".", f"_{ax_name}.")
            yield ft.partial(
                plotDictAsBars,
                group,
                common_meta,
                path,
                getter=getter,
                normalize=self.normalize,
                scale=self.scale,
                style_set=self.style_set,
                show_yields=self.show_yields,
                plot_configuration=pc,
                ax_name=ax_name
            )


@define
class CutflowTable(BasePostprocessor):
    output_name: str
    format: Literal["markdown", "csv", "latex"] = "csv"

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )

        yield ft.partial(
            makeAndSaveCutflowTable,
            group,
            common_meta,
            output_path,
            format=self.format,
        )


def makeCutflowDf(group):
    import pandas as pd

    dataset_cutflows = {}
    cut_order = None
    for selection_flow, metadata in group:
        dataset_cutflows[metadata["dataset_name"]] = _getCutflow(selection_flow)
        if cut_order is None:
            cut_order = list(selection_flow.cuts)
        else:
            if cut_order != list(selection_flow.cuts):
                raise ValueError("Cutflows are not consistent across datasets.")
    all_data = {}
    for dataset_name, cutflow in dataset_cutflows.items():
        all_data[dataset_name, "Events"] = cutflow

    df = pd.DataFrame(all_data)
    for col in df.columns:
        df.loc[:, (col[0], "Eff. Abs.")] = (
            df.loc[:, (col[0], "Events")] / df.loc[:, (col[0], "Events")].iloc[0]
        )
        df.loc[:, (col[0], "Eff. Rel.")] = (
            df.loc[:, (col[0], "Events")] / df.loc[:, (col[0], "Events")].shift(1)
        ).fillna(1)
    df.sort_index(axis=1, level=[0, 1], ascending=[True, False], inplace=True)
    return df


def makeAndSaveCutflowTable(group, common_meta, output_path, format="csv"):
    df = makeCutflowDf(group)
    if format == "csv":
        df.to_csv(output_path)
    elif format == "markdown":
        df.to_markdown(output_path)
    elif format == "latex":
        df.to_latex(output_path)
