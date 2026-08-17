from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
import awkward as ak
from attrs import define, field
from ..common.electrons import CutBasedWPs, cut_mapping as electron_cut_mapping
from ..common.muons import IdWps, IsoWps, cut_mapping as muon_cut_mapping
import numpy as np

import logging

logger = logging.getLogger("analyzer.modules")


@define
class JetID(AnalyzerModule):
    """
    This analyzer creates a jetId column, specifically for newer
    versions of NanoAOD which had it missing due to a bug.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection to be processed.
    output_col: Column
        Column containing the output jetIds.
    Notes
    -----
    - V12 has different recipe than V13, V14, V15 per JME POG.
    - Previous versions don't have this issue and are unchanged.
    - Input collections must be expected to have a jetId column
      in nanoAOD V11 and below to be used as default.
    """

    input_col: Column
    output_col: Column

    def run(self, columns, params):
        metadata = columns.metadata
        nanoversion = metadata["other_data"]["nanoversion"]
        nanoversion = "V" + nanoversion if "V" not in nanoversion else nanoversion
        jets = columns[self.input_col]
        if nanoversion in ["V13", "V14", "V15"]:
            eta = abs(jets.eta)
            jet_id_tight = ak.where(
                eta <= 2.6,
                (jets.neHEF < 0.99)
                & (jets.neEmEF < 0.9)
                & (jets.chMultiplicity + jets.neMultiplicity > 1)
                & (jets.chHEF > 0.01)
                & (jets.chMultiplicity > 0),
                ak.where(
                    (eta > 2.6) & (eta <= 2.7),
                    (jets.neHEF < 0.90) & (jets.neEmEF < 0.99),
                    ak.where(
                        (eta > 2.7) & (eta <= 3.0),
                        (jets.neHEF < 0.99),
                        ak.where(
                            eta > 3.0,
                            (jets.neMultiplicity >= 2) & (jets.neEmEF < 0.4),
                            False,
                        ),
                    ),
                ),
            )

            jet_id_tight_lep_veto = ak.where(
                eta <= 2.7,
                jet_id_tight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),
                jet_id_tight,
            )

            jet_id = ak.where(
                jet_id_tight & jet_id_tight_lep_veto, 6, ak.where(jet_id_tight, 2, 0)
            )
        elif nanoversion == "V12":
            eta = abs(jets.eta)

            jet_id_tight = ak.where(
                eta <= 2.7,
                (jets.jetId & (1 << 1)) > 0,
                ak.where(
                    (eta > 2.7) & (eta <= 3.0),
                    ((jets.jetId & (1 << 1)) > 0) & (jets.neHEF < 0.99),
                    ak.where(
                        eta > 3.0,
                        ((jets.jetId & (1 << 1)) > 0) & (jets.neEmEF < 0.4),
                        False,
                    ),
                ),
            )

            jet_id_tight_lep_veto = ak.where(
                eta <= 2.7,
                jet_id_tight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),
                jet_id_tight,
            )

            jet_id = ak.where(
                jet_id_tight & jet_id_tight_lep_veto, 6, ak.where(jet_id_tight, 2, 0)
            )
        else:
            jet_id = jets.jetId
        columns[self.output_col] = jet_id
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class HElectronMaker(AnalyzerModule):
    """
    Select electrons based on kinematics, cut-based ID, and isolation.

    This analyzer filters electrons in an event according to minimum
    transverse momentum, maximum pseudorapidity, cut-based ID working point,
    and maximum mini-isolation.

    Parameters
    ----------
    input_col : Column
        Column containing the input electron collection.
    output_col : Column
        Column where the selected electrons will be stored.
    working_point : CutBasedWPs
        Cut-based ID working point (fail, veto, loose, medium, tight).
    min_pt : float, optional
        Minimum transverse momentum in GeV, by default 10.
    max_abs_eta : float, optional
        Maximum absolute pseudorapidity, by default 2.4.
    max_abs_dxy: dict, optional
        Dictionary with keys "barrel", "endcap" for dxy selection.
    max_abs_dz: dict, optional
        Dictionary with keys "barrel", "endcap" for dz selection.

    """

    input_col: Column
    output_col: Column
    working_point: CutBasedWPs
    min_pt: float = 10
    max_abs_eta: float = 2.4
    max_abs_dxy: dict = None
    max_abs_dz: dict = None

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        electrons = columns[self.input_col]
        pass_pt = electrons.pt > self.min_pt
        pass_eta = abs(electrons.eta) < self.max_abs_eta
        pass_wp = electrons.cutBased >= electron_cut_mapping[self.working_point]
        if self.max_abs_dxy:
            pass_dxy = abs(electrons.dxy) < ak.where(
                abs(electrons.eta) < 1.479,
                self.max_abs_dxy["barrel"],
                self.max_abs_dxy["endcap"],
            )
        else:
            pass_dxy = True
        if self.max_abs_dz:
            pass_dz = abs(electrons.dxy) < ak.where(
                abs(electrons.eta) < 1.479,
                self.max_abs_dxy["barrel"],
                self.max_abs_dxy["endcap"],
            )
        else:
            pass_dz = True

        columns[self.output_col] = electrons[
            pass_pt & pass_eta & pass_wp & pass_dxy & pass_dz
        ]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class HMuonMaker(AnalyzerModule):
    """
    Select muons based on kinematics, ID, and isolation criteria.

    This analyzer filters muons in an event according to minimum transverse
    momentum, maximum pseudorapidity, a chosen ID working point, and
    optional isolation requirements.

    Parameters
    ----------
    input_col : Column
        Column containing the input muon collection.
    output_col : Column
        Column where the selected muons will be stored.
    id_working_point : IdWps
        Muon ID working point (loose, medium, tight).
    min_pt : float, optional
        Minimum transverse momentum in GeV, by default 10.
    max_abs_eta : float, optional
        Maximum absolute pseudorapidity, by default 2.4.
    iso_working_point : IsoWps or None, optional
        Optional isolation working point. If provided, muons must meet
        the corresponding isolation requirement.
    """

    input_col: Column
    output_col: Column
    id_working_point: IdWps
    min_pt: float = 10
    max_abs_eta: float = 2.4
    iso_working_point: IsoWps | None = None

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        muon = columns[self.input_col]
        pass_pt = muon.pt > self.min_pt
        pass_eta = abs(muon.eta) < self.max_abs_eta
        pass_id_wp = muon[self.id_working_point]
        passed = pass_pt & pass_eta & pass_id_wp
        if self.iso_working_point is not None:
            pass_iso = muon.pfIsoId >= muon_cut_mapping[self.iso_working_point]
            passed = passed & pass_iso

        columns[self.output_col] = muon[passed]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class HJetFilter(AnalyzerModule):
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
    min_btagPNetQvG : float, optional
        Minimum QvG discriminator value for jets, by default 0.0.
    max_abs_eta : float, optional
        Maximum absolute pseudorapidity allowed for jets, by default 2.4.
    include_pu_id : bool, optional
        Whether to apply pileup jet ID requirements (for supported eras),
        by default False.
    include_jet_id : bool, optional
        Whether to apply jet ID requirements, by default False.

    Notes
    -----
    - Jet ID selection requires only the tight bit to be set (bitmask `0b010`).
    - Pileup ID selection is only applied for 2016–2018 eras.
      Jets with pT > 50 GeV automatically pass the PU ID requirement.
    """

    input_col: Column
    output_col: Column
    min_pt: float = 30.0
    min_btagPNetQvG: float = 0.0
    max_abs_eta: float = 2.4
    include_pu_id: bool = False
    include_jet_id: bool = False

    def run(self, columns, params):
        metadata = columns.metadata
        jets = columns[self.input_col]
        good_jets = jets[
            (jets.pt > self.min_pt)
            & (abs(jets.eta) < self.max_abs_eta)
            & (jets.btagPNetQvG > self.min_btagPNetQvG)
        ]

        if self.include_jet_id:
            good_jets = good_jets[((good_jets.jetId & 0b010) != 0)]

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
class TieredPtJetFilter(AnalyzerModule):
    """
    Produces a new jet collection (output_col) from an existing one
    (input_col) using a tiered pt selection: among jets in input_col, the
    n_hard_jets highest-pt jets must individually clear hard_pt_cut; any
    additional jets (ranked below n_hard_jets) only need to clear whatever
    floor input_col already carries. If an event doesn't have n_hard_jets
    genuinely-hard jets, no soft jets are pulled in either -- it degrades to
    exactly a flat hard_pt_cut selection for that event (verified against
    synthetic edge cases; mirrors the tested tiered_pt_selection() in
    nano_to_h5_V2.py / spanet_inputs.py).

    Structured the same way as HJetFilter: operates on the whole coherent
    jet record (columns[input_col]), so every field on the input collection
    -- jetId, puId, hadronFlavour, everything -- carries over automatically,
    not just an explicitly-enumerated subset.

    input_col is expected to already carry a loose pt floor and any eta/ID
    cuts (e.g. HJetFilter with min_pt = soft_pt_cut) -- this module only
    adds the tiered pt logic on top of whatever input_col already contains.

    Parameters
    ----------
    input_col : Column
        The (loosely-selected) input jet collection.
    output_col : Column
        The new tiered-selection jet collection to produce.
    n_hard_jets : int
        Number of highest-pt jets required to clear hard_pt_cut. Default 4.
    hard_pt_cut : float
        pt threshold for the top n_hard_jets. Default 25.0.
    """

    input_col: Column
    output_col: Column
    n_hard_jets: int = 4
    hard_pt_cut: float = 25.0

    def run(self, columns, params):
        jets = columns[self.input_col]

        pt_sort_idx = ak.argsort(jets.pt, axis=1, ascending=False)
        sorted_jets = jets[pt_sort_idx]

        rank = ak.local_index(sorted_jets.pt, axis=1)
        n_hard_in_event = ak.sum(sorted_jets.pt > self.hard_pt_cut, axis=1)
        has_enough_hard = n_hard_in_event >= self.n_hard_jets

        passes_tier = (sorted_jets.pt > self.hard_pt_cut) | (
            has_enough_hard & (rank >= self.n_hard_jets)
        )

        columns[self.output_col] = sorted_jets[passes_tier]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]


@define
class DRVetoFilter(AnalyzerModule):
    """
    Filters an input collection by removing elements that are within a given
    delta R of objects in any of the veto collections.

    Parameters
    ----------
    input_col : Column
        Column containing the collection to be filtered.
    output_col : Column
        Column where the filtered collection will be stored.
    veto_cols : list[Column]
        Columns containing objects to veto against.
    veto_dr : float
        Elements within this delta R of any veto object will be removed.
    """

    input_col: Column
    output_col: Column
    veto_cols: list[Column]
    veto_dr: float

    def run(self, columns, params):
        objects = columns[self.input_col]
        veto_objects = ak.concatenate([columns[col] for col in self.veto_cols], axis=1)
        veto_objects = ak.with_name(veto_objects, "PtEtaPhiMLorentzVector")
        drs = objects.metric_table(veto_objects)
        min_dr = ak.fill_none(ak.min(drs, axis=2), self.veto_dr + 1)
        columns[self.output_col] = objects[min_dr > self.veto_dr]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col] + self.veto_cols

    def outputs(self, metadata):
        return [self.output_col]


@define
class JetCombos(AnalyzerModule):
    """
    Build composite objects from specified combinations
    of jets (by index). Events missing the required number
    of jets are filled with None (i.e. excluded/ignored).

    Parameters
    ----------
    input_cols : list of Column
        Column containing the jet collections.
    jet_combos : list of dict[int: list[int]]
        List of jet combinations. Each entry in the list is a dictionary
        specifying the indices of jets to combine for the corresponding input column.
    output_cols : list[Column]
        Ordered list of Columns for Jets to be stored.
    order_by: list of Column, optional
        List of columns to order the jets by before combining.
        Should be a member of the respective input column.
    ascending: bool, optional
        Whether to sort the jets in ascending order before combining.
    """

    input_cols: list[Column]
    jet_combos: list[dict[int, list[int]]]
    output_cols: list[Column]
    order_by: list[Column] = []
    ascending: bool = False

    def run(self, columns, params):
        for i, combo in enumerate(self.jet_combos):
            sum_cols = []
            combined_msk = None
            for col_idx, rank_idxs in combo.items():
                input_col = self.input_cols[col_idx]
                jets = columns[input_col]
                if self.order_by:
                    order_col = columns[input_col + self.order_by[col_idx]]
                    jets = jets[ak.argsort(order_col, axis=1, ascending=self.ascending)]
                max_idx = max(rank_idxs)
                padded = ak.pad_none(jets, max_idx + 1, axis=1)
                msk = ak.num(jets, axis=1) < max_idx + 1
                sum_cols.append(padded[:, rank_idxs])
                if combined_msk is None:
                    combined_msk = msk
                else:
                    combined_msk = combined_msk | msk

            combined_col = ak.concatenate(sum_cols, axis=1)
            summed = combined_col.sum()
            summed = ak.mask(summed, ~combined_msk)
            columns[self.output_cols[i]] = ak.with_name(
                ak.zip(
                    {
                        "pt": ak.fill_none(summed.pt, np.nan),
                        "eta": ak.fill_none(summed.eta, np.nan),
                        "phi": ak.fill_none(summed.phi, np.nan),
                        "mass": ak.fill_none(summed.mass, np.nan),
                    }
                ),
                "PtEtaPhiMLorentzVector",
            )
        return columns, []

    def outputs(self, metadata):
        return self.output_cols

    def inputs(self, metadata):
        return self.input_cols + self.order_by


@define
class AbsoluteValue(AnalyzerModule):
    """
    This simple module takes the absolute value of a specified input column and
    stores the result in a new output column.
    """

    input_col: Column
    output_col: Column

    def run(self, columns, params):
        columns[self.output_col] = abs(columns[self.input_col])
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
