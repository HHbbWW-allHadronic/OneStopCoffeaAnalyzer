from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define, field, evolve
from ..common.axis import RegularAxis
from ..common.histogram_builder import makeHistogram


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
        factory=lambda: RegularAxis(20, 0, 1000, "", unit="GeV")
    )

    def run(self, columns, params):
        jets = columns[self.input_col]
        ret = []
        mass_axis = evolve(self.mass_axis, name=f"{self.hist_name} $m$")
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

