import awkward as ak
from attrs import define
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column


@define
class FatBQuarkMaker(AnalyzerModule):
    """
    Select boosted FatJets based on regressed mass and Xbb discriminant thresholds.

    Parameters
    ----------
    input_col : Column
        Column containing the input FatJet collection.
    output_col : Column
        Column where the selected FatJets will be stored.
    min_mass : float
        Minimum regressed mass threshold (default: 50.0).
    max_mass : float
        Maximum regressed mass threshold (default: 250.0).
    min_xbb : float
        Minimum Xbb discriminant score threshold (default: 0.8).
    mass_branch : str
        Field name for regressed mass on the FatJet collection (default: "massRegressed").
    xbb_branch : str
        Field name for the Xbb score on the FatJet collection (default: "xbb").
    """

    input_col: Column
    output_col: Column
    min_mass: float = 50.0
    max_mass: float = 250.0
    min_xbb: float = 0.8
    mass_branch: str = "massRegressed"
    xbb_branch: str = "xbb"

    def run(self, columns, params):
        fatjets = columns[self.input_col]

        mass = fatjets[self.mass_branch]
        xbb_score = fatjets[self.xbb_branch]
        eta = fatjets.eta
        pt = fatjets.pt

        # Mask for FatJets with 50 <= massRegressed <= 250 and Xbb >= 0.8
        mask = (
            (mass >= self.min_mass)
            & (mass <= self.max_mass)
            & (xbb_score >= self.min_xbb)
            & (abs(eta) <= 2.4)
            & (pt >= 300)
        )

        selected_fatjets = fatjets[mask]
        columns[self.output_col] = selected_fatjets

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
