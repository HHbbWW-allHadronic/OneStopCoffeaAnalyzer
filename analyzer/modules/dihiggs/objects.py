from analyzer.core.analysis_modules import AnalyzerModule
import re

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import flatten
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import itertools as it
from attrs import define, field, evolve
from ..common.axis import RegularAxis
from ..common.histogram_builder import makeHistogram
from ..common.electrons import CutBasedWPs, cut_mapping as electron_cut_mapping
from ..common.muons import IdWps, IsoWps, cut_mapping as muon_cut_mapping
import enum

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
    max_abs_eta: float = 2.4
    include_pu_id: bool = False
    include_jet_id: bool = False

    def run(self, columns, params):
        metadata = columns.metadata
        jets = columns[self.input_col]
        good_jets = jets[(jets.pt > self.min_pt) & (abs(jets.eta) < self.max_abs_eta)]

        if self.include_jet_id:
            good_jets = good_jets[
                ((good_jets.jetId & 0b010) != 0)
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

    """

    input_col: Column
    output_col: Column
    working_point: CutBasedWPs
    min_pt: float = 10
    max_abs_eta: float = 2.4

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        electrons = columns[self.input_col]
        pass_pt = electrons.pt > self.min_pt
        pass_eta = abs(electrons.eta) < self.max_abs_eta
        pass_wp = electrons.cutBased >= electron_cut_mapping[self.working_point]
        columns[self.output_col] = electrons[pass_pt & pass_eta & pass_wp]
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
