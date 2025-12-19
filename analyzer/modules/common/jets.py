from analyzer.core.analysis_modules import AnalyzerModule, register_module

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
import awkward as ak
import itertools as it
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
import copy
from rich import print
import numpy as np


import awkward as ak
import correctionlib
import pydantic as pyd
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import correctionlib.schemav2 as cs
from functools import lru_cache


from analyzer.core.analysis_modules import (
    AnalyzerModule,
    register_module,
    MetadataExpr,
    MetadataAnd,
    IsRun,
    IsSampleType,
)


@define
class FilterNear(AnalyzerModule):
    target_col: Column
    near_col: Column
    output_col: Column
    max_dr: float

    def run(self, columns, params):
        target = columns[self.target_col]
        near = columns[self.near_col]
        nearest = target.nearest(near, threshold=self.max_dr)
        filtered = target[ak.is_none(nearest, axis=1)]

        columns[self.output_col] = filtered
        return columns, []

    def inputs(self, metadata):
        return [self.target_col, self.near_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class PromoteIndex(AnalyzerModule):
    input_col: Column
    output_col: Column
    index: int = 0

    def run(self, columns, params):
        columns[self.output_col] = columns[self.input_col][:, self.index]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class VetoMapFilter(AnalyzerModule):
    input_col: Column
    output_col: Column
    # should_run: MetadataFunc = field(factory=lambda: IS_RUN_2)
    should_run: MetadataExpr = field(factory=lambda: IsRun(2))
    veto_type = "jetvetomap"

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        corr = self.getCorr(columns.metadata)
        j = columns[self.input_col]
        j = j[
            (abs(j.eta) < 2.4)
            & (j.pt > 15)
            & ((j.jetId & 0b100) != 0)
            & ((j.chEmEF + j.neEmEF) < 0.9)
        ]
        vetoes = corr.evaluate(self.veto_type, j.eta, j.phi)
        passed_jets = j[(vetoes == 0)]
        columns[self.output_col] = passed_jets
        return columns, []

    def getCorr(self, metadata):
        info = metadata["era"]["jet_veto_map"]
        path, name = info["file"], info["name"]
        k = (path, name)
        if k in self.__corrections:
            return self.__corrections[k]
        ret = correctionlib.CorrectionSet.from_file(path)[name]
        self.__corrections[k] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorr(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class VetoMap(AnalyzerModule):
    input_col: Column
    selection_name: str = "jet_veto_map"
    # should_run: MetadataFunc = field(factory=lambda: IS_RUN_2)
    should_run: MetadataExpr = field(factory=lambda: IsRun(3))

    def run(self, columns, params):
        map_name = columns.metadata["era"]["jet_veto_map"]["name"]
        corr = self.getCorr(columns.metadata)[map_name]
        j = columns[self.input_col]
        j = j[
            (abs(j.eta) < 2.4)
            & (j.pt > 15)
            & ((j.jetId & 0b100) != 0)
            & ((j.chEmEF + j.neEmEF) < 0.9)
        ]
        vetoes = eval_veto.evaluate(veto_type, j.eta, j.phi)
        passed = ak.any(vetoes != 0, axis=1)
        addSelection(columns, self.selection_name, passed)
        return columns, []

    def getCorr(self, metadata):
        file_path = metadata["era"]["jet_veto_map"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)
        self.__corrections[file_path] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorr(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]


@define
class JetEtaPhiVeto(AnalyzerModule):
    input_col: Column
    phi_range: tuple[float, float]
    eta_range: tuple[float, float]
    run_range: tuple[float, float] | None = None
    selection_name: str = "jet_eta_phi_veto"

    def run(self, columns, params):
        j = columns[self.input_col]
        j = j[
            (abs(j.eta) < 2.4)
            & (j.pt > 15)
            & ((j.jetId & 0b100) != 0)
            & ((j.chEmEF + j.neEmEF) < 0.9)
        ]

        in_phi = (j.phi > self.phi_range[0]) & (j.phi < self.phi_range[1])
        in_eta = (j.eta > self.eta_range[0]) & (j.eta < self.eta_range[1])

        if self.run_range is None:
            any_in = ak.any(in_phi & in_eta, axis=1)
        else:
            veto_run = columns["run"]
            any_in = ak.any(in_phi & in_eta, axis=1) & veto_run

        addSelection(columns, self.selection_name, ~any_in)
        return columns, []

    def inputs(self, metadata):
        if self.run_range is None:
            return [self.input_col]
        else:
            return [self.input_col, Column("run")]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]


@define
class HT(AnalyzerModule):
    input_col: Column
    output_col: Column = field(factory=lambda: Column("HT"))

    def run(self, columns, params):
        jets = columns[self.input_col]
        columns[self.output_col] = ak.sum(jets.pt, axis=1)
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class JetFilter(AnalyzerModule):
    input_col: Column
    output_col: Column
    min_pt: float = 30.0
    max_abs_eta: float = 2.4
    include_pu_id: bool = False
    include_jet_id: bool = False

    def run(self, columns, params):
        metadata = columns.metadata
        jets = columns[self.input_col]
        good_jets = jets[(jets.pt > self.min_pt) & (abs(jets.eta) < self.max_abs_eta)]

        if self.include_jet_id:
            good_jets = good_jets[
                ((good_jets.jetId & 0b100) != 0) & ((good_jets.jetId & 0b010) != 0)
            ]

        if self.include_pu_id:
            if any(
                x in metadata["era"]["name"]
                for x in ["2016", "2017", "2018"]
            ):
                good_jets = good_jets[(good_jets.pt > 50) | ((good_jets.puId & 0b10) != 0)]
        columns[self.output_col] = good_jets
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class TopJetHistograms(AnalyzerModule):
    prefix: str
    input_col: Column
    max_idx: int

    def run(self, columns, params):
        jets = columns[self.input_col]
        ret = []
        padded = ak.pad_none(jets, self.max_idx, axis=1)
        for i in range(0, self.max_idx):
            mask = ak.num(jets, axis=1) > i
            ret.append(
                makeHistogram(
                    f"{self.prefix}_pt_{i+1}",
                    columns,
                    RegularAxis(20, 0, 1000, f"$p_{{T, {i+1}}}$", unit="GeV"),
                    padded[:, i].pt,
                    description=f"$p_T$ of jet {i+1} ",
                    mask=mask,
                )
            )
            ret.append(
                makeHistogram(
                    f"{self.prefix}_eta_{i+1}",
                    columns,
                    RegularAxis(20, -4, 4, f"$\\eta_{{T, {i+1}}}$"),
                    padded[:, i].eta,
                    description=f"$\\eta$ of jet {i+1} ",
                    mask=mask,
                )
            )
            ret.append(
                makeHistogram(
                    f"{self.prefix}_phi_{i+1}",
                    columns,
                    RegularAxis(20, -4, 4, f"$\\phi_{{T, {i+1}}}$"),
                    padded[:, i].phi,
                    description=f"$\\phi$ of jet {i+1} ",
                    mask=mask,
                )
            )

        return columns, ret

    def outputs(self, metadata):
        return []

    def inputs(self, metadata):
        return [self.input_col]


@define
class JetComboHistograms(AnalyzerModule):
    prefix: str
    input_col: Column
    jet_combos: list[tuple]

    def run(self, columns, params):
        jets = columns[self.input_col]
        ret = []
        padded = ak.pad_none(jets, self.max_idx, axis=1)
        for combo in self.combos:
            mask = ak.num(jets, axis=1) > i
            jets = gj[:, i:j].sum()
            masses[(i, j)] = jets.mass
            ret.append(
                makeHistogram(
                    f"{self.prefix}_pt_{i+1}",
                    columns,
                    RegularAxis(20, 0, 3000, f"$p_{{T, {i+1}}}$", unit="GeV"),
                    padded[:, i].pt,
                    description=f"$p_T$ of jet {i+1} ",
                    mask=mask,
                )
            )

        return columns, ret

    def outputs(self, metadata):
        return []

    def inputs(self, metadata):
        return [self.input_col]


@define
class JetScaleCorrections(AnalyzerModule):
    input_col: Column
    output_col: Column
    jet_type: str = "AK4"
    use_regrouped: bool = True

    @staticmethod
    def getKeyJec(name, jet_type, params):
        jec_params = params["dataset"]["era"]["jet_corrections"]
        jet_type = jec_params.jet_names[jet_type]
        data_mc = "MC" if params.dataset.sample_type == "MC" else "DATA"
        campaign = jec_params.jec.campaign
        version = jec_params.jec.version
        return f"{campaign}_{version}_{data_mc}_{name}_{jet_type}"

    def run(self, columns, params):
        metadata = columns.metadata

        jets = columns[self.input_col]
        corrections = self.getCorrection(metadata)
        systematics = params["jec_systematic"]

        if systematic == "central":
            columns[self.output_col] = jets
            return columns, []

        pt_raw = (1 - jets.rawFactor) * jets.pt
        mass_raw = (1 - jets.rawFactor) * jets.mass
        rho = (
            columns["fixedGridRhoFastjetAll"]
            if "fixedGridRhoFastjetAll" in columns.fields
            else columns["Rho.fixedGridRhoFastjetAll"]
        )

        k = JetScaleCorrections.getKeyJec(systematic, jet_type, params)
        evaluator = corrections[k]
        factor = to_f32(evaluator.evaluate(jets.eta, pt_raw))
        shift = -1 if systematic.startswith("down") else 1
        corrected = ak.with_field(jets, toF32(jets.pt * (1.0 + factor * shift)), "pt")
        corrected = ak.with_field(
            corrected, toF32(jets.mass * (1.0 + factor * shift)), "mass"
        )
        columns[self.output_col] = corrected
        return columns, []

    def getCorrection(self, metadata):
        file_path = metadata["era"]["jet_corrections"]["file"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)
        self.__corrections[file_path] = ret
        return ret

    def getParameterSpec(self, metadata):
        if use_regrouped:
            systematics = metadata["jec_params"]["jec"]["regrouped_systematics"]
        else:
            systematics = metadata["jec_params"]["jec"]["systematics"]

        possible_values = it.product(["up", "down"], systematics)
        possible_values = ["central"] + [
            f"{updown}_jes{name}" for updown, name in possible_values
        ]
        return ModuleParameterSpec(
            {
                "jes_variation": ParameterSpec(
                    default_value="central",
                    possible_values=possible_values,
                    tags={
                        "shape_variation",
                        "jes",
                    },
                ),
            }
        )

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class JetResolutionCorrections(AnalyzerModule):
    input_col: Column
    output_col: Column
    jet_type: str = "AK4"

    should_run: MetadataExpr = field(factory=lambda: IsSampleType(MC))

    @staticmethod
    def getKeyJer(name, jet_type, params):
        jet_type = params["jet_names"][jet_type]
        data_mc = "MC" if params.dataset.sample_type == "MC" else "DATA"
        campaign = jec_params["jer"]["campaign"]
        version = jec_params["jer"]["version"]
        return f"{campaign}_{version}_{data_mc}_{name}_{jet_type}"

    def run(self, columns, params):
        metadata = columns.metadata
        jec_params = dataset["era"]["jet_corrections"]
        jets = columns[self.input_col]
        systematic = params["systematic"]

        corrections = self.getCorrection(metadata)
        res_key = JetResolutionCorrections.getKeyJer(
            "PtResolution", self.jet_type, metadata["era"]["jet_corrections"]
        )
        sf_key = JetResolutionCorrections.getKeyJer(
            "ScaleFactor", self.jet_type, metadata["era"]["jet_corrections"]
        )
        eval_res = correction[res_key]
        eval_sf = correction[sf_key]
        eval_smear = getSmearer(metadata)

        rho = (
            columns["fixedGridRhoFastjetAll"]
            if "fixedGridRhoFastjetAll" in columns.fields
            else columns["Rho", "fixedGridRhoFastjetAll"]
        )

        inputs = {
            "JetEta": jets.eta,
            "JetPt": jets.pt,
            "Rho": rho,
            "systematic": systematic,
        }
        jer = eval_res.evaluate(*[inputs[inp.name] for inp in evaljer.inputs])
        jer = toF32(jer)

        matched_genjet = jets.matched_gen
        gen_idxs = jets[genjet_idx_col]
        valid_gen_idxs = ak.mask(gen_idxs, (gen_idxs >= 0) & (gen_idxs < 10))
        padded_gen_jets = ak.pad_none(gen_jets, 10, axis=1)
        matched_genjet = padded_gen_jets[valid_gen_idxs]
        pt_relative_diff = 1 - matched_genjet.pt / jets.pt
        is_matched_pt = abs(pt_relative_diff) < (3 * jer)

        smear_inputs = inputs | {
            "JER": jer,
            "JERSF": sf,
            "GenPt": ak.fill_none(is_matched_pt * matched_genjet.pt, -1),
            "EventID": ak.values_astype(rho * 10**6, "int64"),
        }

        final_smear = evalsmear.evaluate(
            *[smear_inputs[inp.name] for inp in evalsmear.inputs]
        )
        final_smear = toF32(final_smear)
        smeared_jets = ak.with_field(jets, toF32(jets.pt * final_smear), "pt")
        smeared_jets = ak.with_field(
            smeared_jets, toF32(jets.mass * final_smear), "mass"
        )
        columns[self.output_col] = smeared_jets
        return columns, []

    def getSmearer(self, metadata):
        file_path = metadata["era"]["jet_corrections"]["files"]["smear"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)["JERSmear"]
        self.__corrections[file_path] = ret
        return ret

    def getCorrection(self, metadata):
        file_path = metadata["era"]["jet_corrections"]["file"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)
        self.__corrections[file_path] = ret
        return ret

    def getParameterSpec(self, metadata):
        if use_regrouped:
            systematics = metadata["jec_params"]["jec"]["regrouped_systematics"]
        else:
            systematics = metadata["jec_params"]["jec"]["systematics"]

        possible_values = it.product(["up", "down"], systematics)
        possible_values = ["central"] + [
            f"{updown}_jes{name}" for updown, name in possible_values
        ]
        return ModuleParameterSpec(
            {
                "jes_variation": ParameterSpec(
                    default_value="central",
                    possible_values=possible_values,
                    tags={
                        "shape_variation",
                        "jes",
                    },
                ),
            }
        )

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)
        self.getSmearer(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


def corrected_jets(
    columns,
    params,
    input_col: str = None,
    output_col: str = None,
    jet_type="AK4",
    do_smearing=False,
    include_systematics=True,
    use_regrouped=True,
):
    jec_params = params.dataset.era.jet_corrections

    if use_regrouped:
        systematics = jec_params.jec.regrouped_systematics
    else:
        systematics = jec_params.jec.systematics

    jets = columns[input_col]

    corrections_path = jec_params.files[jet_type]
    # cset = correctionlib.CorrectionSet.from_file(corrections_path)

    pt_raw = (1 - jets.rawFactor) * jets.pt
    mass_raw = (1 - jets.rawFactor) * jets.mass
    rho = (
        columns.fixedGridRhoFastjetAll
        if "fixedGridRhoFastjetAll" in columns.fields
        else columns.Rho.fixedGridRhoFastjetAll
    )

    systs = {}

    if include_systematics:
        for systematic in systematics:
            k = getKeyJec(systematic, jet_type, params)
            logger.info(f"Getting jet correction key {k}")
            corr = getEvaluator(corrections_path, k)
            # event_rho = getRho(events, jec_params.rho_name)
            factor = to_f32(corr.evaluate(jets.eta, pt_raw))
            for shift_name, shift in [("up", 1), ("down", -1)]:
                # fields = {field: jets[field] for field in jets.fields}

                corrected = jets
                # corrected["pt"] = jets.pt * (1.0 + factor * shift)
                # corrected["mass"] = jets.mass * (1.0 + factor * shift)
                corrected = ak.with_field(
                    corrected, to_f32(jets.pt * (1.0 + factor * shift)), "pt"
                )
                corrected = ak.with_field(
                    corrected, to_f32(jets.mass * (1.0 + factor * shift)), "mass"
                )

                if do_smearing:
                    corrected = smearJets(
                        corrected, columns.GenJet, rho, params, jet_type
                    )

                systematic_name = f"{shift_name}_jes{systematic}"
                logger.info(f"Adding jet systematic {systematic_name}")

                systs[systematic_name] = corrected

    if do_smearing:
        rjets = smearJets(
            jets,
            columns.GenJet,
            rho,
            params,
            jet_type,
            include_systematics=include_systematics,
        )
        systs_jer = {}
        if include_systematics:
            rjets, systs_jer = rjets
    else:
        rjets = jets
        systs_jer = {}

    columns.add(output_col, rjets, systs | systs_jer | {"base": jets})


@define
class PileupJetIdSF(AnalyzerModule):
    input_col: Column
    weight_name: str = "puid_sf"
    should_run: MetadataExpr = field(factory=lambda: IsSampleType(MC))

    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="nom",
                    possible_values=["nom", "up", "down"],
                    tags={"weight_variation"},
                ),
            }
        )

    def run(self, columns, params):
        eval_pu = self.getCorrection(columns.metadata)

        jets = columns[self.input_col]
        pu_jets = jets[jets.pt < 50]
        matched_pujet = pu_jets[pu_jets.genJetIdx > -1]

        inputs_matched = {
            "eta": matched_pujet.eta,
            "pt": matched_pujet.pt,
            "systematic": "nom",
            "workingpoint": working_point,
        }
        sf_matched = eval_pu.evaluate(
            *[inputs_matched[inp.name] for inp in eval_pu.inputs]
        )
        sf = ak.prod(sf_matched, axis=1)

        columns["Weights", self.weight_name] = sf
        return columns, []

    def getCorrection(self, metadata):
        puidinfo = metadata["era"]["jet_pileup_id"]
        file_path = puidinfo["file"]
        name = puidinfo["name"]
        if (name, file_path) in self.__corrections:
            return self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = cset[name]
        self.__corrections[(name, file_path)] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Weights", self.weight_name))]
