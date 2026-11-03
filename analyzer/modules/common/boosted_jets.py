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

        fatjets["globalParT3_Top"] = p_top
        fatjets["THVV"] = thvv

        columns[self.output_col] = fatjets

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]







@define
class HHKinematicVarsMaker(AnalyzerModule):
    """
    Computes di-fatjet (HH = bb + VV) kinematics and ratios for BDT input:
    - m_jj: Invariant mass of (bb + VV)
    - pt_jj: Transverse momentum of (bb + VV)
    - dPhi_jj: |dPhi(bb, VV)|
    - ratio_ptVV_ptbb: pt(VV) / pt(bb)
    - ratio_ptVV_ptjj: pt(VV) / pt(jj)
    """

    bb_col: Column
    vv_col: Column
    output_col: Column

    def run(self, columns, params):
        bb_jets = columns[self.bb_col]
        vv_jets = columns[self.vv_col]

        # Extract 1D record per event (returns None for events with 0 candidates)
        bb = ak.firsts(bb_jets)
        vv = ak.firsts(vv_jets)

        # Di-fatjet 4-vector sum
        hh = bb + vv

        # Compute scalar quantities (these contain None for empty events)
        m_jj = hh.mass
        pt_jj = hh.pt
        dphi_jj = abs(bb.delta_phi(vv))

        bb_pt = bb.pt
        vv_pt = vv.pt

        # Safe denominators prevent divide-by-zero RuntimeWarnings
        safe_bb_pt = ak.where((bb_pt > 0) & (~ak.is_none(bb_pt)), bb_pt, 1.0)
        safe_pt_jj = ak.where((pt_jj > 0) & (~ak.is_none(pt_jj)), pt_jj, 1.0)

        ratio_ptVV_ptbb = ak.where(
            (bb_pt > 0) & (~ak.is_none(bb_pt)), vv_pt / safe_bb_pt, 0.0
        )
        ratio_ptVV_ptjj = ak.where(
            (pt_jj > 0) & (~ak.is_none(pt_jj)), vv_pt / safe_pt_jj, 0.0
        )

        # Convert option arrays (None) to var-length collections ([]) using ak.singletons
        # so HistogramBuilder can flatten them naturally
        columns[self.output_col] = ak.zip(
            {
                "m_jj": ak.singletons(m_jj),
                "pt_jj": ak.singletons(pt_jj),
                "dPhi_jj": ak.singletons(dphi_jj),
                "ratio_ptVV_ptbb": ak.singletons(ratio_ptVV_ptbb),
                "ratio_ptVV_ptjj": ak.singletons(ratio_ptVV_ptjj),
            }
        )

        return columns, []

    def inputs(self, metadata):
        return [self.bb_col, self.vv_col]

    def outputs(self, metadata):
        return [self.output_col]

