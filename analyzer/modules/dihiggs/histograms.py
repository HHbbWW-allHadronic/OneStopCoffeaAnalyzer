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
        factory=lambda: RegularAxis(20, 0, 200, "", unit="GeV")
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

@define
class JetComboHistogram2D(AnalyzerModule):
    """
    Build composite objects from specified combinations
    of jets (by index) and produce 2D histogram of their 
    invariant mass. Histograms are filled only for events 
    where all required jets are present.

    Parameters
    ----------
    prefixes : list of str
        Prefix used for naming the generated histogram.
    input_cols : list of Column
        Column containing the jet collection.
    jet_combos : list of list of int
        List of jet index combinations. Each inner list specifies the
        indices of jets to be combined (e.g. ``[0, 1]`` for the leading
        two jets). Number of jet combos should be equal to input columns.
    mass_axes : list[RegularAxis]
        Axes to use for the mass plotting.
    """

    prefixes: list[str]
    input_cols: list[Column]
    jet_combos: list[list[int]]
    mass_axes: list[RegularAxis]

    def run(self, columns, params):
        ret = []
        masks = []
        masses = []
        axes = []
        names = f"{self.prefixes[0]}_{self.prefixes[1]}_m2d"

        for i, input_col in enumerate(self.input_cols):
            combo = self.jet_combos[i]
            jets = columns[input_col]
            max_idx = max(combo)
            padded = ak.pad_none(jets, max_idx + 1, axis=1)
            mask = ak.num(jets, axis=1) > max_idx
            masks.append(mask)

            summed = padded[:, combo].sum()
            masses.append(summed.mass)

            axis = self.mass_axes[i]
            name_suffix = f"$m_{{{''.join(str(x) for x in combo)}}}$"

            if axis.name:
                new_name = f"{axis.name} {name_suffix}"
            else:
                new_name = name_suffix

            axis = evolve(axis, name=new_name)
            axes.append(axis)

        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask & mask

        ret.append(
            makeHistogram(
                names, 
                columns,
                axes,
                masses,
                mask=combined_mask,
            )
        )

        return columns, ret

    def outputs(self, metadata):
        return []

    def inputs(self, metadata):
        return self.input_cols
