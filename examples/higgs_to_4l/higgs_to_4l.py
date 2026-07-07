"""
Rough replication of the H->4L analysis as implemented in
https://github.com/cms-opendata-analyses/HiggsToFourLeptonsNanoAODOutreachAnalysis
"""

from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column, addSelection
from attrs import define
from typing import Literal
import awkward as ak

Z_MASS = 91.1876


@define
class Hto4LExampleGoodMuonMaker(AnalyzerModule):
    input_col: Column = Column("Muon")
    output_col: Column = Column("GoodMuon")

    min_pt: float = 5.0
    max_abs_eta: float = 2.4
    max_iso: float = 0.40
    max_dxy: float = 0.5
    max_dz: float = 1.0
    max_sip3d: float = 4.0

    def run(self, columns, params):
        muon = columns[self.input_col]
        pass_pt = muon.pt > self.min_pt
        pass_eta = abs(muon.eta) < self.max_abs_eta
        pass_iso = abs(muon.pfRelIso04_all) < self.max_iso
        muon_ip3d = (muon.dxy**2 + muon.dz**2) ** 0.5
        muon_sip3d = muon_ip3d / ((muon.dxyErr**2 + muon.dzErr**2) ** 0.5)
        pass_dxy = abs(muon.dxy) < self.max_dxy
        pass_dz = abs(muon.dz) < self.max_dz
        pass_sip3d = muon_sip3d < self.max_sip3d
        passed = pass_pt & pass_eta & pass_iso & pass_dxy & pass_dz & pass_sip3d
        columns[self.output_col] = muon[passed]

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class Hto4LExampleGoodElectronMaker(AnalyzerModule):
    input_col: Column = Column("Electron")
    output_col: Column = Column("GoodElectron")

    min_pt: float = 7.0
    max_abs_eta: float = 2.5
    max_iso: float = 0.40
    max_dxy: float = 0.5
    max_dz: float = 1.0
    max_sip3d: float = 4.0

    def run(self, columns, params):
        electron = columns[self.input_col]

        pass_pt = electron.pt > self.min_pt
        pass_eta = abs(electron.eta) < self.max_abs_eta
        pass_iso = abs(electron.pfRelIso03_all) < self.max_iso

        electron_ip3d = (electron.dxy**2 + electron.dz**2) ** 0.5
        electron_sip3d = electron_ip3d / (
            (electron.dxyErr**2 + electron.dzErr**2) ** 0.5
        )

        pass_dxy = abs(electron.dxy) < self.max_dxy
        pass_dz = abs(electron.dz) < self.max_dz
        pass_sip3d = electron_sip3d < self.max_sip3d

        passed = pass_pt & pass_eta & pass_iso & pass_dxy & pass_dz & pass_sip3d
        columns[self.output_col] = electron[passed]

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class Hto4LExampleApplyZZScaleFactor(AnalyzerModule):
    weight_name: str = "zz_scale_factor"

    def run(self, columns, params):
        other_data = columns.metadata.get("other_data", {})
        scale_factor = other_data.get("scaleFactorZZTo4l", 1.0)

        if "run" in columns.fields:
            weight_col = ak.full_like(columns["run"], scale_factor, dtype=float)
        else:
            weight_col = scale_factor

        columns["Weights", self.weight_name] = weight_col
        return columns, []

    def inputs(self, metadata):
        return [Column("run")]

    def outputs(self, metadata):
        return [Column(("Weights", self.weight_name))]


@define
class Hto4LExampleHiggsTo4lSelection(AnalyzerModule):
    channel: Literal["4mu", "4e", "2e2mu"]
    muon_col: Column = Column("GoodMuon")
    electron_col: Column = Column("GoodElectron")
    selection_name: str = "FourLeptonSelection"

    def run(self, columns, params):
        if self.channel == "4mu":
            muons = columns[self.muon_col]
            sum_charge = ak.sum(muons.charge, axis=1) == 0
            sel = sum_charge

        elif self.channel == "4e":
            electrons = columns[self.electron_col]
            sum_charge = ak.sum(electrons.charge, axis=1) == 0
            sel = sum_charge

        elif self.channel == "2e2mu":
            muons = columns[self.muon_col]
            electrons = columns[self.electron_col]

            # pad to evaluate cuts without awkward broadcasting errors on empty arrays
            mu_padded = ak.pad_none(muons, 2)
            el_padded = ak.pad_none(electrons, 2)

            pt_cut = ((mu_padded[:, 0].pt > 20) & (mu_padded[:, 1].pt > 10)) | (
                (el_padded[:, 0].pt > 20) & (el_padded[:, 1].pt > 10)
            )

            charge_cut = (ak.sum(muons.charge, axis=1) == 0) & (
                ak.sum(electrons.charge, axis=1) == 0
            )

            sel = ak.fill_none(pt_cut, False) & charge_cut

        else:
            raise ValueError(f"Unknown channel: {self.channel}")

        addSelection(columns, self.selection_name, ak.fill_none(sel, False))
        return columns, []

    def inputs(self, metadata):
        inputs = []
        if self.channel in ["4mu", "2e2mu"]:
            inputs.append(self.muon_col)
        if self.channel in ["4e", "2e2mu"]:
            inputs.append(self.electron_col)
        return inputs

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]


def buildHiggs4l(leptons):
    PAIRS = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]
    l = [leptons[:, i : i + 1] for i in range(4)]

    def isOS(l0, l1):
        return l0.charge != l1.charge

    def isSep(l0, l1, dr_cut=0.02):
        return l0.delta_r(l1) >= dr_cut

    za = ak.concatenate([l[i] + l[j] for i, j, _, _ in PAIRS], axis=1)
    zb = ak.concatenate([l[k] + l[m] for _, _, k, m in PAIRS], axis=1)

    valid_os_1 = ak.concatenate([isOS(l[i], l[j]) for i, j, _, _ in PAIRS], axis=1)
    valid_os_2 = ak.concatenate([isOS(l[k], l[m]) for _, _, k, m in PAIRS], axis=1)
    valid_os = valid_os_1 & valid_os_2

    valid_dr_1 = ak.concatenate([isSep(l[i], l[j]) for i, j, _, _ in PAIRS], axis=1)
    valid_dr_2 = ak.concatenate([isSep(l[k], l[m]) for _, _, k, m in PAIRS], axis=1)
    valid_dr = valid_dr_1 & valid_dr_2

    closer_to_z = abs(za.mass - Z_MASS) < abs(zb.mass - Z_MASS)
    z1 = ak.where(closer_to_z, za, zb)
    z2 = ak.where(closer_to_z, zb, za)

    z1_mass_dev = ak.where(valid_os, abs(z1.mass - Z_MASS), 99999.0)
    best = ak.argmin(z1_mass_dev, axis=1, keepdims=True)

    best_z1 = ak.firsts(z1[best])
    best_z2 = ak.firsts(z2[best])
    best_valid = ak.firsts(valid_os[best]) & ak.firsts(valid_dr[best])

    return best_z1, best_z2, best_z1 + best_z2, best_valid


def buildHiggs2e2mu(muons, electrons):
    m0, m1 = muons[:, 0], muons[:, 1]
    e0, e1 = electrons[:, 0], electrons[:, 1]

    zm, ze = m0 + m1, e0 + e1

    md_m = abs(zm.mass - Z_MASS)
    md_e = abs(ze.mass - Z_MASS)

    best_z1 = ak.where(md_m < md_e, zm, ze)
    best_z2 = ak.where(md_m < md_e, ze, zm)
    higgs = best_z1 + best_z2
    return best_z1, best_z2, higgs


@define
class Hto4LExampleHiggsTo4lReco(AnalyzerModule):
    channel: Literal["4mu", "4e", "2e2mu"]
    muon_col: Column = Column("GoodMuon")
    electron_col: Column = Column("GoodElectron")

    def run(self, columns, params):
        if self.channel in ["4mu", "4e"]:
            lep_col = self.muon_col if self.channel == "4mu" else self.electron_col
            leps = columns[lep_col]
            z1, z2, higgs, valid = buildHiggs4l(leps)

        elif self.channel == "2e2mu":
            muons = columns[self.muon_col]
            electrons = columns[self.electron_col]
            z1, z2, higgs = buildHiggs2e2mu(muons, electrons)
            valid = ak.ones_like(higgs.mass, dtype=bool)

        z_mass_cut = (z1.mass > 40) & (z1.mass < 120) & (z2.mass > 12) & (z2.mass < 120)
        final_valid = valid & ak.fill_none(z_mass_cut, False)

        columns["Higgs_mass"] = higgs.mass
        columns["Z1_mass"] = z1.mass
        columns["Z2_mass"] = z2.mass

        addSelection(columns, "ReconstructHiggs", ak.fill_none(final_valid, False))

        return columns, []

    def inputs(self, metadata):
        inputs = []
        if self.channel in ["4mu", "2e2mu"]:
            inputs.append(self.muon_col)
        if self.channel in ["4e", "2e2mu"]:
            inputs.append(self.electron_col)
        return inputs

    def outputs(self, metadata):
        return [
            Column("Higgs_mass"),
            Column("Z1_mass"),
            Column("Z2_mass"),
            Column(("Selection", "ReconstructHiggs")),
        ]
