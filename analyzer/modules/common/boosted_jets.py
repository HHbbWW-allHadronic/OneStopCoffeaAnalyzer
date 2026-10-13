from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define
import awkward as ak
from typing import Optional


@define
class GlobalParT3XbbDiscriminator(AnalyzerModule):
    """
    Computes the mass-decorrelated GlobalParT-3 X->bb discriminator score:
    D = sig / (sig + bkg)
    """

    input_col: Column
    output_col: Column
    sig_branch: str = "globalParT3_Xbb"
    qcd_branch: str = "globalParT3_QCD"

    def run(self, columns, params):
        jets = columns[self.input_col]

        sig = jets[self.sig_branch]
        bkg = jets[self.qcd_branch]


        denom = sig + bkg

        # Compute ratio safely to prevent 0/0 division NaNs
        disc = ak.where(denom > 0, sig / denom, 0.0)

        columns[self.output_col] = disc

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]



@define
class RegressedMass(AnalyzerModule):
    """
    m_regressed = massCorrGeneric * mass * (1 - rawFactor)
    """

    input_col: Column
    output_col: Column 
    corr_branch: str = "globalParT3_massCorrGeneric"
    mass_branch: str = "mass"
    raw_factor_branch: str = "rawFactor"

    def run(self, columns, params):
        jets = columns[self.input_col]

        corr = jets[self.corr_branch]
        mass = jets[self.mass_branch]
        raw_factor = jets[self.raw_factor_branch]

        reg_mass = corr * mass * (1.0 - raw_factor)
        columns[self.output_col] = reg_mass

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]



@define
class THVVMaker(AnalyzerModule):
    """
    Computes GlobalParT-3 H->VV discriminant score:
    T_HVV = (P_HVV4q + P_HVV3q) / (P_QCD + P_Top + P_HVV4q + P_HVV3q)
    """

    input_col: Column
    output_col: Column

    def run(self, columns, params):
        fatjets = columns[self.input_col]

        p_top = (
            fatjets.globalParT3_TopbWev
            + fatjets.globalParT3_TopbWmv
            + fatjets.globalParT3_TopbWq
            + fatjets.globalParT3_TopbWqq
            + fatjets.globalParT3_TopbWtauhv
        )

        num = fatjets.globalParT3_XWW4q + fatjets.globalParT3_XWW3q

        den = fatjets.globalParT3_QCD + p_top + num

        thvv = ak.where(den > 0, num / den, 0.0)

        fatjets["THVV"] = thvv
        columns[self.output_col] = fatjets

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
