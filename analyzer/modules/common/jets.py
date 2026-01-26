from analyzer.core.analysis_modules import AnalyzerModule
import re

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import flatten
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import itertools as it
from attrs import define, field, evolve
from .axis import RegularAxis
from .histogram_builder import makeHistogram


import correctionlib
import logging


from analyzer.core.analysis_modules import (
    MetadataExpr,
    MetadataAnd,
    IsRun,
    IsSampleType,
)

logger = logging.getLogger("analyzer.modules")


@define
class FilterNear(AnalyzerModule):
    r"""
    Filter objects based on proximity to another collection.

    This analyzer removes entries from a target collection that are within
    a specified angular distance ($\Delta R$) of objects in a second collection.
    Objects in the target collection are kept only if **no nearby object**
    is found within the given distance threshold.
    Proximity is determined using the ``nearest`` method on the target collection.

    Parameters
    ----------
    target_col : Column
        Column containing the collection to be filtered.
    near_col : Column
        Column containing the collection of nearby reference objects.
    output_col : Column
        Column where the filtered target collection will be stored.
    max_dr : float
        Maximum $\Delta R$ distance used to define proximity.

    """

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
class VetoMapFilter(AnalyzerModule):
    r"""
    Apply a detector veto map to an input jet collection
    in order to remove jets falling into problematic detector regions.
    The veto map is evaluated as a function of jet ($\eta$, $\phi$) coordinates.


    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    output_col : Column
        Column where the filtered jet collection will be stored.
    should_run : MetadataExpr, optional
        Expression controlling whether this module should run for a given
        dataset based on metadata. By default, this filter runs only for
        Run 2 datasets.
    veto_type : str
        Identifier passed to the correction evaluator to select the
        appropriate veto map. Defaults to ``"jetvetomap"``.
    """

    input_col: Column
    output_col: Column
    # should_run: MetadataFunc = field(factory=lambda: IS_RUN_2)
    should_run: MetadataExpr = field(factory=lambda: IsRun(2))
    veto_type: str = "jetvetomap"

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
    r"""
    Event-level veto selection based on detector veto maps.
    Events are excluded if any jet passing a loose selection lies withing a vetoed $\eta$-$\phi$  region.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    selection_name : str, optional
        Name of the event-level selection to be added to the column
        collection, by default ``"jet_veto_map"``.
    should_run : MetadataExpr, optional
        Expression controlling whether this module should run for a given
        dataset based on metadata. By default, this filter runs only for
        Run 3 datasets.

    """

    input_col: Column
    selection_name: str = "jet_veto_map"
    # should_run: MetadataFunc = field(factory=lambda: IS_RUN_2)
    should_run: MetadataExpr = field(factory=lambda: IsRun(3))
    veto_type: str = "jetvetomap"
    __corrections: dict = field(factory=dict)

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
        vetoes = corr.evaluate(self.veto_type, j.eta, j.phi)
        failed = ak.any(vetoes != 0, axis=1)
        addSelection(columns, self.selection_name, ~failed)
        return columns, []

    def getCorr(self, metadata):
        info = metadata["era"]["jet_veto_map"]
        path, name = info["file"], info["name"]
        k = (path, name)
        if k in self.__corrections:
            return self.__corrections[k]
        ret = correctionlib.CorrectionSet.from_file(path)
        self.__corrections[k] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorr(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]


@define
class JetEtaPhiVeto(AnalyzerModule):
    r"""
    Event-level veto based on jet ($\eta$, $\phi$) regions.

    This analyzer defines an event-level selection that vetoes events
    containing jets within a specified rectangular region in (η, φ)
    space. The veto can optionally be restricted to a specific run
    range.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    phi_range : tuple of float
        Inclusive (min, max) range in phi defining the veto region.
    eta_range : tuple of float
        Inclusive (min, max) range in eta defining the veto region.
    run_range : tuple of float or None, optional
        Optional run range over which the veto should be applied. If
        ``None``, the veto is applied to all runs.
    selection_name : str, optional
        Name of the event-level selection to be added, by default
        ``"jet_eta_phi_veto"``.

    """

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
            return [self.input_col, Column(("run",))]

    def outputs(self, metadata):
        return [Column(("Selection", self.selection_name))]


@define
class HT(AnalyzerModule):
    """
    Compute the scalar sum of jet transverse momenta (H_T).

    Parameters
    ----------
    input_col : Column
        Column containing the jet collection whose pT values will be
        summed.
    output_col : Column, optional
        Column where the computed H_T values will be stored. By default,
        a column named ``"HT"`` is created.

    """

    input_col: Column
    output_col: Column = field(factory=lambda: Column(("HT",)))

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
    """
    This analyzer filters an input jet collection according to transverse
    momentum and pseudorapidity requirements, with optional jet ID and pileup
    ID selections. The resulting filtered jet collection is written to a new
    output column.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection to be filtered.
    output_col : Column
        Column where the filtered jet collection will be stored.
    min_pt : float, optional
        Minimum transverse momentum (pT) threshold for jets, by default 30.0.
    max_abs_eta : float, optional
        Maximum absolute pseudorapidity allowed for jets, by default 2.4.
    include_pu_id : bool, optional
        Whether to apply pileup jet ID requirements (for supported eras),
        by default False.
    include_jet_id : bool, optional
        Whether to apply jet ID requirements, by default False.

    Notes
    -----
    - Jet ID selection requires both the tight and tightLepVeto bits
      to be set (bitmask `0b100` and `0b010`).
    - Pileup ID selection is only applied for 2016–2018 eras.
      Jets with pT > 50 GeV automatically pass the PU ID requirement.
    """

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
            if any(x in metadata["era"]["name"] for x in ["2016", "2017", "2018"]):
                good_jets = good_jets[
                    (good_jets.pt > 50) | ((good_jets.puId & 0b10) != 0)
                ]
        columns[self.output_col] = good_jets
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class TopVecHistograms(AnalyzerModule):
    r"""
    Produce kinematic histograms for the leading objects in a collection.

    This analyzer creates histograms of $p_T$,$\eta$, and $\phi$ for the first *N* objects
    in a vector-like collection (e.g. jets), where *N* is configurable.
    Histograms are produced only for events where the corresponding
    object exists.

    Parameters
    ----------
    prefix : str
        Prefix used for naming the generated histograms.
    input_col : Column
        Column containing the object collection (e.g. jets).
    max_idx : int
        Maximum number of leading objects for which histograms are
        produced.
    """

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
                    f"{self.prefix}_pt_{i + 1}",
                    columns,
                    RegularAxis(20, 0, 1000, f"$p_{{T, {i + 1}}}$", unit="GeV"),
                    padded[:, i].pt,
                    description=f"$p_T$ of jet {i + 1} ",
                    mask=mask,
                )
            )
            ret.append(
                makeHistogram(
                    f"{self.prefix}_eta_{i + 1}",
                    columns,
                    RegularAxis(20, -4, 4, f"$\\eta_{{T, {i + 1}}}$"),
                    padded[:, i].eta,
                    description=f"$\\eta$ of jet {i + 1} ",
                    mask=mask,
                )
            )
            ret.append(
                makeHistogram(
                    f"{self.prefix}_phi_{i + 1}",
                    columns,
                    RegularAxis(20, -4, 4, f"$\\phi_{{T, {i + 1}}}$"),
                    padded[:, i].phi,
                    description=f"$\\phi$ of jet {i + 1} ",
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
    """
    Build composite objects from specified combinations
    of jets (by index) and produces histograms of their invariant mass
    and transverse momentum. Each combination is treated independently,
    and histograms are filled only for events where all required jets
    are present.

    Parameters
    ----------
    prefix : str
        Prefix used for naming the generated histograms.
    input_col : Column
        Column containing the jet collection.
    jet_combos : list of list of int
        List of jet index combinations. Each inner list specifies the
        indices of jets to be combined (e.g. ``[0, 1]`` for the leading
        two jets).
    """

    prefix: str
    input_col: Column
    jet_combos: list[list[int]]
    mass_axis: RegularAxis = field(
        factory=lambda: RegularAxis(50, 0, 3000, "", unit="GeV")
    )

    def run(self, columns, params):
        jets = columns[self.input_col]
        ret = []
        max_idx = max(flatten(self.jet_combos))
        padded = ak.pad_none(jets, max_idx + 1, axis=1)
        for combo in self.jet_combos:
            i, j = min(combo), max(combo)
            mask = ak.num(jets, axis=1) > max(combo)
            summed = padded[:, combo].sum()

            axis = self.mass_axis
            name_suffix = f"$m_{{{i + 1}{j + 1}}}$"

            if axis.name:
                new_name = f"{axis.name} {name_suffix}"
            else:
                new_name = name_suffix

            axis = evolve(axis, name=new_name)

            ret.append(
                makeHistogram(
                    f"{self.prefix}_{i + 1}{j + 1}_m",
                    columns,
                    axis,
                    summed.mass,
                    mask=mask,
                )
            )
            ret.append(
                makeHistogram(
                    f"{self.prefix}_{i + 1}{j + 1}_pt",
                    columns,
                    RegularAxis(50, 0, 3000, f"$pt_{{{i + 1}{j + 1}}}$", unit="GeV"),
                    summed.pt,
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
    """
    This analyzer adjusts jet transverse momentum (pT) and mass
    according to jet energy corrections (JEC) derived from
    calibration campaigns. Both nominal ("central") and systematic
    variations (up/down JES) are supported. Corrections are applied
    per jet based on its kinematics and dataset metadata.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    output_col : Column
        Column where the corrected jets will be stored.
    jet_type : str, optional
        Jet type for which corrections are applied (e.g. ``"AK4"``),
        by default ``"AK4"``.
    use_regrouped : bool, optional
        Whether to use regrouped JES systematics from metadata
        (recommended), by default ``True``.

    """

    input_col: Column
    output_col: Column
    jet_type: str = "AK4"
    use_regrouped: bool = True

    __corrections: dict = field(factory=dict)

    @staticmethod
    def getKeyJec(name, jet_type, metadata):
        jec_params = metadata["era"]["jet_corrections"]
        jet_type = jec_params["jet_names"][jet_type]
        data_mc = "MC" if metadata["sample_type"] == "MC" else "DATA"
        campaign = jec_params["jec"]["campaign"]
        version = jec_params["jec"]["version"]
        ret = f"{campaign}_{version}_{data_mc}_{name}_{jet_type}"
        logger.debug(f'Using JEC Key "{ret}"')
        return ret

    def run(self, columns, params):
        metadata = columns.metadata
        jets = columns[self.input_col]
        corrections = self.getCorrection(metadata)
        systematic = params["variation"]
        real_syst_name = re.sub(r"(up|down)_jes", "", systematic)
        logger.debug(
            f'Running JEC with systematic "{systematic}". Real name is "{real_syst_name}"'
        )

        if systematic == "central":
            columns[self.output_col] = jets
            return columns, []

        pt_raw = (1 - jets.rawFactor) * jets.pt
        (1 - jets.rawFactor) * jets.mass
        (
            columns["fixedGridRhoFastjetAll"]
            if "fixedGridRhoFastjetAll" in columns.fields
            else columns["Rho.fixedGridRhoFastjetAll"]
        )

        k = JetScaleCorrections.getKeyJec(real_syst_name, self.jet_type, metadata)
        evaluator = corrections[k]
        factor = evaluator.evaluate(jets.eta, pt_raw)
        shift = -1 if systematic.startswith("down") else 1
        corrected = ak.with_field(jets, (jets.pt * (1.0 + factor * shift)), "pt")
        corrected = ak.with_field(
            corrected, (jets.mass * (1.0 + factor * shift)), "mass"
        )
        columns[self.output_col] = corrected
        return columns, []

    def getCorrection(self, metadata):
        file_path = metadata["era"]["jet_corrections"]["files"][self.jet_type]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)
        self.__corrections[file_path] = ret
        return ret

    def getParameterSpec(self, metadata):
        jec_meta = metadata["era"]["jet_corrections"]["jec"]
        if self.use_regrouped:
            systematics = jec_meta["regrouped_systematics"]
        else:
            systematics = jec_meta["systematics"]

        possible_values = it.product(["up", "down"], systematics)
        possible_values = ["central"] + [
            f"{updown}_jes{name}" for updown, name in possible_values
        ]
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
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
        return [
            self.input_col,
            Column("fixedGridRhoFastjetAll"),
            Column("Rho.fixedGridRhoFastjetAll"),
        ]

    def outputs(self, metadata):
        return [self.output_col]


@define
class JetResolutionCorrections(AnalyzerModule):
    """
    This analyzer adjusts jet transverse momentum (pT) and mass to account
    for detector resolution effects in simulated (MC) events. Smearing is
    applied based on the per-jet resolution, matching to generated jets,
    and optional systematic variations.

    Parameters
    ----------
    input_col : Column
        Column containing the reconstructed jet collection.
    genjet_col : Column
        Column containing the generator-level jets used for matching.
    output_col : Column
        Column where the smeared jets will be stored.
    jet_type : str, optional
        Jet type (e.g., ``"AK4"``) for which JER corrections are applied,
        by default ``"AK4"``.
    use_regrouped : bool, optional
        Whether to use regrouped JER systematics, by default ``True``.
    should_run : MetadataExpr, optional
        Expression controlling whether this module should run (default: MC
        events only).

    Notes
    -----
    - Jets are matched to gen-jets via metadata-specified columns.
    - Correction sets are cached per file to avoid repeated I/O.
    """

    input_col: Column
    genjet_col: Column
    output_col: Column
    jet_type: str = "AK4"

    use_regrouped: bool = True
    should_run: MetadataExpr = field(factory=lambda: IsSampleType("MC"))

    __corrections: dict = field(factory=dict)

    @staticmethod
    def getKeyJer(name, jet_type, metadata):
        jet_params = metadata["era"]["jet_corrections"]
        jet_type = jet_params["jet_names"][jet_type]
        data_mc = "MC" if metadata["sample_type"] == "MC" else "DATA"
        campaign = jet_params["jer"]["campaign"]
        version = jet_params["jer"]["version"]
        ret = f"{campaign}_{version}_{data_mc}_{name}_{jet_type}"
        logger.debug(f'Using JER Key "{ret}"')
        return ret

    def run(self, columns, params):
        metadata = columns.metadata
        jet_params = metadata["era"]["jet_corrections"]
        genjet_idx_col = jet_params["jer"]["genjet_idx_col"]
        jets = columns[self.input_col]
        systematic = params["variation"].replace("_JER", "")
        corrections = self.getCorrection(metadata)
        res_key = JetResolutionCorrections.getKeyJer(
            "PtResolution", self.jet_type, metadata
        )
        sf_key = JetResolutionCorrections.getKeyJer(
            "ScaleFactor", self.jet_type, metadata
        )
        eval_res = corrections[res_key]
        eval_sf = corrections[sf_key]
        eval_smear = self.getSmearer(metadata)

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

        jer = eval_res.evaluate(*[inputs[inp.name] for inp in eval_res.inputs])
        sf_inputs = inputs | {"systematic": systematic}
        sf = eval_sf.evaluate(*[sf_inputs[inp.name] for inp in eval_sf.inputs])

        gen_jets = columns[self.genjet_col]
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

        final_smear = eval_smear.evaluate(
            *[smear_inputs[inp.name] for inp in eval_smear.inputs]
        )
        final_smear = final_smear
        smeared_jets = ak.with_field(jets, (jets.pt * final_smear), "pt")
        smeared_jets = ak.with_field(smeared_jets, (jets.mass * final_smear), "mass")
        columns[self.output_col] = smeared_jets
        return columns, []

    def getParameterSpec(self, metadata):
        jer_meta = metadata["era"]["jet_corrections"]["jer"]
        systematics = jer_meta["systematics"]
        possible_values = ["nom"] + [f"{updown}_JER" for updown in systematics]
        return ModuleParameterSpec(
            {
                "variation": ParameterSpec(
                    default_value="nom",
                    possible_values=possible_values,
                    tags={
                        "shape_variation",
                        "jer",
                    },
                ),
            }
        )

    def getSmearer(self, metadata):
        file_path = metadata["era"]["jet_corrections"]["files"]["smear"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)["JERSmear"]
        self.__corrections[file_path] = ret
        return ret

    def getCorrection(self, metadata):
        file_path = metadata["era"]["jet_corrections"]["files"][self.jet_type]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        ret = correctionlib.CorrectionSet.from_file(file_path)
        self.__corrections[file_path] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)
        self.getSmearer(metadata)

    def inputs(self, metadata):
        return [
            self.input_col,
            self.genjet_col,
            Column("fixedGridRhoFastjetAll"),
            Column("Rho.fixedGridRhoFastjetAll"),
        ]

    def outputs(self, metadata):
        return [self.output_col]


@define
class PileupJetIdSF(AnalyzerModule):
    """
    Compute pileup jet ID scale factors for Monte Carlo events.

    This analyzer calculates per-event weights corresponding to the
    pileup jet ID efficiency correction. Only jets with pT < 50 GeV
    and matched to a generator-level jet are considered.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    working_point : str
        Pileup ID working point to use (e.g., ``"loose"``, ``"medium"``,
        ``"tight"``).
    weight_name : str, optional
        Name of the weight column to store, by default ``"puid_sf"``.
    should_run : MetadataExpr, optional
        Expression controlling whether this module should run. By default,
        runs only for Run 2 Monte Carlo datasets.

    Notes
    -----
    - The per-jet scale factors are combined multiplicatively per event.
    - Uses correction files defined in dataset metadata under
      ``metadata["era"]["jet_pileup_id"]``.
    """

    input_col: Column
    working_point: str
    weight_name: str = "puid_sf"
    should_run: MetadataExpr = field(
        factory=lambda: MetadataAnd([IsSampleType("MC"), IsRun(2)])
    )

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
            "workingpoint": self.working_point,
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
        key = (name, file_path)
        if key in self.__corrections:
            return self.__corrections[key]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = cset[name]
        self.__corrections[key] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Weights", self.weight_name))]
