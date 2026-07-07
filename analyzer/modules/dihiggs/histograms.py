from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define, field, evolve
from ..common.axis import RegularAxis
from ..common.histogram_builder import makeHistogram
import awkward as ak

import correctionlib
import logging


@define
class FourVecHistograms(AnalyzerModule):
    r"""
    Produce kinematic histograms for jet-like columns.
    This analyzer creates histograms of $p_T$, $\eta$, mass, and $\phi$.

    Parameters
    ----------
    input_col : Column
        Column containing the object collection (e.g. jets).
    hist_name: str
        Name of column to be used in histogram.
    mass_axis:
        RegularAxis for mass plotting.
    """

    input_col: Column
    hist_name: str
    mass_axis: RegularAxis = field(
        factory=lambda: RegularAxis(20, 0, 200, "", unit="GeV")
    )

    def run(self, columns, params):
        jets = columns[self.input_col]
        ret = []
        if axis.name:
            new_name = axis.name
        else:
            new_name = f"{self.hist_name} $m$"
        mass_axis = evolve(self.mass_axis, name=new_name)
        ret.append(
            makeHistogram(
                f"{self.hist_name}_pt",
                columns,
                RegularAxis(20, 0, 1000, f"{self.hist_name} $p_{{T}}$", unit="GeV"),
                jets.pt,
                description=f"$p_T$ of {self.hist_name}",
            )
        )
        ret.append(
            makeHistogram(
                f"{self.hist_name}_eta",
                columns,
                RegularAxis(20, -4, 4, f"{self.hist_name} $\\eta$"),
                jets.eta,
                description=f"$\\eta$ of {self.hist_name}",
            )
        )
        ret.append(
            makeHistogram(
                f"{self.hist_name}_phi",
                columns,
                RegularAxis(20, -4, 4, f"{self.hist_name} $\\phi$"),
                jets.phi,
                description=f"$\\phi$ of {self.hist_name}",
            )
        )
        ret.append(
            makeHistogram(
                f"{self.hist_name}_mass",
                columns,
                mass_axis,
                jets.mass,
            )
        )

        return columns, ret

    def outputs(self, metadata):
        return []

    def inputs(self, metadata):
        return [self.input_col]


@define
class JetVarRankHistograms(AnalyzerModule):
    """
    Produce histograms of a variable for the first N jets in an event.

    Parameters
    ----------
    hist_name: str
        Name of column to be used in histogram. Can be formatted for 2D histograms with second variable names.
    input_col : Column
        Column containing the object collection (e.g. jets).
    axis: RegularAxis
        Axis for the variable to be plotted.
    second_names: list[str]
        Names of the second variable to be plotted in 2D histograms (for hist name).
    second_cols: list[Column]
        Columns containing the second variable to be plotted in 2D histograms.
    second_axes: list[RegularAxis]
        Axes for the second variable to be plotted in 2D histograms.
    max_idx: int
        Largest jet index to be plotted (default is 6).
    """

    hist_name: str
    input_col: Column
    axis: RegularAxis
    second_names: list[str] = []
    second_cols: list[Column] = []
    second_axes: list[RegularAxis] = []
    max_idx: int = 6

    def run(self, columns, params):
        var = columns[self.input_col]
        ret = []
        padded = ak.pad_none(var, self.max_idx, axis=1)
        for i in range(0, self.max_idx):
            mask = ak.num(var, axis=1) > i
            jet_individual = padded[:, i]
            rank_label = f"$_{{{i + 1}}}$"
            new_name = f"{self.axis.name} {rank_label}"

            axis = evolve(self.axis, name=new_name)
            # Generate the histogram for Jet [i+1]
            if not self.second_names:
                ret.append(
                    makeHistogram(
                        f"{self.hist_name}{i+1}",
                        columns,
                        axis,
                        jet_individual,
                        description=f"{self.axis.name} of jet {i + 1}",
                        mask=mask,
                    )
                )
            else:
                for j in range(len(self.second_names)):
                    hist_name = self.hist_name.format(f"{i+1}_{self.second_names[j]}")
                    second_var = columns[self.second_cols[j]]
                    second_axis = self.second_axes[j]
                    ret.append(
                        makeHistogram(
                            hist_name,
                            columns,
                            [axis, second_axis],
                            [jet_individual, second_var],
                            description=f"2d {self.axis.name} of jet {i + 1} and {second_axis.name}",
                            mask=mask,
                        )
                    )
        return columns, ret

    def outputs(self, metadata):
        return []

    def inputs(self, metadata):
        return [self.input_col] + self.second_cols
