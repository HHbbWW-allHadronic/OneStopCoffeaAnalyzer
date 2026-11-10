import awkward as ak
from attrs import define
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column


@define
class FatJetMaker(AnalyzerModule):
    """
    Select FatJets based on regressed mass and Xbb discriminant thresholds,
    basic kinematic cuts.

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
    min_xbb : float, optional
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
    min_xbb: float | None = None
    mass_branch: str = "massRegressed"
    xbb_branch: str = "xbbDiscrim"

    def run(self, columns, params):
        fatjets = columns[self.input_col]

        mass = fatjets[self.mass_branch]
        xbb_score = fatjets[self.xbb_branch]
        eta = fatjets.eta
        pt = fatjets.pt

        # Mask for FatJets with 50 <= massRegressed <= 250 and Xbb >= min_xbb
        mask = (
            (mass >= self.min_mass)
            & (mass <= self.max_mass)
            & (xbb_score >= self.min_xbb if self.min_xbb is not None else True)
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


@define
class HHCandidateSelector(AnalyzerModule):
    """
    Selects HH -> bbVV jet candidates from an input jet collection:
    1. 'bb_output_col': Jet with highest Xbb score, requiring Xbb >= min_xbb.
    2. 'vv_output_col': Jet with highest THVV score among remaining jets, ONLY IF
       a valid bb-candidate was found in the event.

    Parameters
    ----------
    input_col : Column
        Column containing the pre-selected boosted FatJet collection.
    bb_output_col : Column
        Column where the selected bb-candidate FatJet will be stored.
    vv_output_col : Column
        Column where the selected VV-candidate FatJet will be stored.
    min_xbb : float
        Minimum Xbb score threshold for the bb-candidate (default: 0.8).
    xbb_branch : str
        Field name for Xbb score (default: "xbbDiscrim").
    thvv_branch : str
        Field name for THVV score (default: "THVV").
    """

    input_col: Column
    bb_output_col: Column
    vv_output_col: Column
    min_xbb: float = 0.8
    xbb_branch: str = "xbbDiscrim"
    thvv_branch: str = "THVV"

    def run(self, columns, params):
        fatjets = columns[self.input_col]

        # 1. Sort full jet collection by Xbb score (descending)
        xbb_score = fatjets[self.xbb_branch]
        bb_sorted_idx = ak.argsort(xbb_score, axis=1, ascending=False)
        bb_sorted_jets = fatjets[bb_sorted_idx]

        top_bb = bb_sorted_jets[:, :1]

        bb_pass_mask = top_bb[self.xbb_branch] >= self.min_xbb
        bb_candidate = top_bb[bb_pass_mask]

        has_valid_bb = ak.num(bb_candidate, axis=1) > 0

        remaining_jets = bb_sorted_jets[:, 1:]

        thvv_score = remaining_jets[self.thvv_branch]
        vv_sorted_idx = ak.argsort(thvv_score, axis=1, ascending=False)
        vv_sorted_jets = remaining_jets[vv_sorted_idx]

        top_vv = vv_sorted_jets[:, :1]

        vv_candidate = ak.where(has_valid_bb, top_vv, top_vv[:, :0])

        columns[self.bb_output_col] = bb_candidate
        columns[self.vv_output_col] = vv_candidate

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.bb_output_col, self.vv_output_col]
