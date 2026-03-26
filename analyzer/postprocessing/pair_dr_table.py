from __future__ import annotations
import functools as ft
from typing import Literal
import pandas as pd
import numpy as np
from .style import StyleSet
from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    dotFormat,
)
from .processors import BasePostprocessor
from attrs import define, field
from pathlib import Path

PAIR_LABELS_15 = [
    "b1-b2",
    "b1-q1", "b1-q2", "b1-q3", "b1-q4",
    "b2-q1", "b2-q2", "b2-q3", "b2-q4",
    "q1-q2 (same W on)",
    "q1-q3 (cross W)", "q1-q4 (cross W)",
    "q2-q3 (cross W)", "q2-q4 (cross W)",
    "q3-q4 (same W off)",
]

PAIR_LABELS_6 = [
    "q1-q2 (same W on)",
    "q3-q4 (same W off)",
    "q1-q3 (cross W)",
    "q1-q4 (cross W)",
    "q2-q3 (cross W)",
    "q2-q4 (cross W)",
]


def makeAndSavePairDRTable(group, common_meta, output_path, format="csv"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Accumulate counts across all items in group
    all_counts = None
    for item, meta in group:
        h = item.histogram
        counts = h.values()[0]
        if all_counts is None:
            all_counts = counts.copy()
        else:
            all_counts = all_counts + counts

    # Infer labels from number of bins
    n_bins = len(all_counts)
    if n_bins == 15:
        pair_labels = PAIR_LABELS_15
    elif n_bins == 6:
        pair_labels = PAIR_LABELS_6
    else:
        raise ValueError(f"Unexpected number of bins: {n_bins}. Expected 6 or 15.")

    total = sum(all_counts)
    sorted_pairs = sorted(
        zip(pair_labels, all_counts),
        key=lambda x: -x[1]
    )
    df = pd.DataFrame(
        [
            {
                "Rank": rank,
                "Pair": label,
                "Count": count,
                "Frequency (%)": f"{100 * count / total:.1f}%",
            }
            for rank, (label, count) in enumerate(sorted_pairs, start=1)
        ]
    )
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "markdown":
        df.to_markdown(output_path, index=False)
    elif format == "latex":
        df.to_latex(output_path, index=False)


@define
class PairDRTable(BasePostprocessor):
    output_name: str
    format: Literal["markdown", "csv", "latex"] = "csv"

    def getRunFuncs(self, group, prefix=None):
        common_meta = commonDict(group)
        output_path = dotFormat(
            self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
        )
        yield ft.partial(
            makeAndSavePairDRTable,
            group,
            common_meta,
            output_path,
            format=self.format,
        )
