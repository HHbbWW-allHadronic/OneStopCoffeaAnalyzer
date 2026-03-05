from analyzer.core.analysis_modules import AnalyzerModule
import re

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import flatten
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import numpy as np
import itertools as it
from attrs import define, field, evolve
import enum

import logging


from analyzer.core.analysis_modules import (
    MetadataExpr,
    MetadataAnd,
    IsRun,
    IsSampleType,
)


logger = logging.getLogger("analyzer.modules")

@define
class GenPartMinDRMaker(AnalyzerModule):
    """
    Compute the minimum delta R among all unique pairs of gen quarks.
    
    Parameters
    ----------
    input_col : Column
        Column containing the GenPart collection (must have eta, phi fields).
    output_col : Column
        Column where the per-event minimum delta R scalar will be stored.
    """
    input_col: Column
    output_col: Column

    def run(self, columns, params):
        gen_parts = columns[self.input_col]
        pairs = ak.combinations(gen_parts, 2, axis=1)
        qi, qj = ak.unzip(pairs)

        deta = qi.eta - qj.eta
        dphi = qi.phi - qj.phi
        dphi = ak.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
        dphi = ak.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
        dr_all = np.sqrt(deta**2 + dphi**2)

        columns[self.output_col] = ak.min(dr_all, axis=1)
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

class HHSampleType(enum.Enum):
    SIGNAL = "signal"
    TTBAR_HADRONIC = "ttbar_hadronic"
    TTBAR_SEMILEPTONIC = "ttbar_semileptonic"


@define
class GenPartDecayWalker(AnalyzerModule):
    """
    Walk the GenPart decay tree to find the 6 (or 4 for ttbar semileptonic)
    first-copy quarks from HH->bbWW->bbqqqq or tt->bbWW->bbqqqq decays.

    Parameters
    ----------
    input_col : Column
        Column containing the GenPart collection.
    output_col : Column
        Column where the filtered quark collection will be stored.
    sample_type : HHSampleType
        The sample type, which determines the decay tree traversal path.
        Options: SIGNAL, TTBAR_HADRONIC, TTBAR_SEMILEPTONIC
    """
    input_col: Column
    output_col: Column
    sample_type: HHSampleType

    def _has_ancestor_with(self, idx, mother_idxs, pids, target_pid, target_status=None):
        """
        Walk up the decay tree from idx to check if any ancestor matches
        target_pid and optionally target_status.
        mother_idxs and pids are flat numpy arrays for a single event.
        """
        visited = set()
        current = idx
        while current >= 0 and current not in visited:
            visited.add(current)
            if abs(pids[current]) == target_pid:
                if target_status is None:
                    return True
                # status check would need to be passed in — see note below
                return True
            current = mother_idxs[current]
        return False

    def _walk_signal(self, gen_parts):
        """
        For signal HH->bbWW->bbqqqq:
        Find status 23 quarks whose ancestry includes a status 62 Higgs (PID 25).
        """
        import numpy as np

        result_mask_events = []

        # Work event by event since we need to walk the tree
        mother_idxs = ak.to_list(gen_parts.genPartIdxMother)
        pdg_ids = ak.to_list(gen_parts.pdgId)
        statuses = ak.to_list(gen_parts.status)

        for ev in range(len(mother_idxs)):
            ev_mother = mother_idxs[ev]
            ev_pid = pdg_ids[ev]
            ev_status = statuses[ev]
            n = len(ev_pid)

            selected = []
            for i in range(n):
                # Must be status 23 and a quark we care about
                if ev_status[i] != 23:
                    continue
                if abs(ev_pid[i]) not in {5, 1, 2, 3, 4}:
                    continue

                # Walk up to find a status 62 Higgs ancestor
                visited = set()
                current = ev_mother[i]
                found_higgs = False
                while current >= 0 and current not in visited:
                    visited.add(current)
                    if abs(ev_pid[current]) == 25 and ev_status[current] == 62:
                        found_higgs = True
                        break
                    current = ev_mother[current]

                if found_higgs:
                    selected.append(i)

            if len(selected) != 6:
                raise ValueError(
                    f"Signal event {ev}: expected 6 quarks, found {len(selected)}. "
                    f"Check decay chain integrity."
                )
            result_mask_events.append(selected)

        return result_mask_events

    def _walk_ttbar_hadronic(self, gen_parts):
        """
        For TTbar hadronic tt->bbWW->bbqqqq:
        Find status 23 quarks whose ancestry includes a PID ±24 W
        which itself comes from a PID ±6 top.
        """
        mother_idxs = ak.to_list(gen_parts.genPartIdxMother)
        pdg_ids = ak.to_list(gen_parts.pdgId)
        statuses = ak.to_list(gen_parts.status)

        result_mask_events = []

        for ev in range(len(mother_idxs)):
            ev_mother = mother_idxs[ev]
            ev_pid = pdg_ids[ev]
            ev_status = statuses[ev]
            n = len(ev_pid)

            selected = []
            for i in range(n):
                if ev_status[i] != 23:
                    continue
                if abs(ev_pid[i]) not in {5, 1, 2, 3, 4}:
                    continue

                # Walk up — for b quarks, expect direct top ancestor
                # For light quarks, expect W ancestor whose mother is a top
                visited = set()
                current = ev_mother[i]
                found_top_ancestry = False

                while current >= 0 and current not in visited:
                    visited.add(current)
                    if abs(ev_pid[current]) == 6:
                        found_top_ancestry = True
                        break
                    current = ev_mother[current]

                if found_top_ancestry:
                    selected.append(i)

            if len(selected) != 6:
                raise ValueError(
                    f"TTbar hadronic event {ev}: expected 6 quarks, found {len(selected)}. "
                    f"Check decay chain integrity."
                )
            result_mask_events.append(selected)

        return result_mask_events

    def _walk_ttbar_semileptonic(self, gen_parts):
        """
        For TTbar semileptonic:
        Collect b quarks from both tops, but only light quarks from the
        hadronic W. Skip the leptonic W daughters entirely.
        Expected result: 4 quarks (2b + 2 light)
        """
        mother_idxs = ak.to_list(gen_parts.genPartIdxMother)
        pdg_ids = ak.to_list(gen_parts.pdgId)
        statuses = ak.to_list(gen_parts.status)

        result_mask_events = []

        for ev in range(len(mother_idxs)):
            ev_mother = mother_idxs[ev]
            ev_pid = pdg_ids[ev]
            ev_status = statuses[ev]
            n = len(ev_pid)

            selected = []
            for i in range(n):
                if ev_status[i] != 23:
                    continue

                pid = abs(ev_pid[i])

                # Collect b quarks with top ancestry
                if pid == 5:
                    visited = set()
                    current = ev_mother[i]
                    found_top = False
                    while current >= 0 and current not in visited:
                        visited.add(current)
                        if abs(ev_pid[current]) == 6:
                            found_top = True
                            break
                        current = ev_mother[current]
                    if found_top:
                        selected.append(i)

                # Collect light quarks only from hadronic W
                # (leptonic W daughters will be leptons/neutrinos, not in 1-4)
                elif pid in {1, 2, 3, 4}:
                    visited = set()
                    current = ev_mother[i]
                    found_w_from_top = False
                    while current >= 0 and current not in visited:
                        visited.add(current)
                        if abs(ev_pid[current]) == 24:
                            # Check this W comes from a top
                            w_mother = ev_mother[current]
                            if w_mother >= 0 and abs(ev_pid[w_mother]) == 6:
                                found_w_from_top = True
                            break
                        current = ev_mother[current]
                    if found_w_from_top:
                        selected.append(i)

            if len(selected) != 4:
                raise ValueError(
                    f"TTbar semileptonic event {ev}: expected 4 quarks, found {len(selected)}. "
                    f"Check decay chain integrity."
                )
            result_mask_events.append(selected)

        return result_mask_events

    def run(self, columns, params):
        gen_parts = columns[self.input_col]

        if self.sample_type == HHSampleType.SIGNAL:
            selected_indices = self._walk_signal(gen_parts)
        elif self.sample_type == HHSampleType.TTBAR_HADRONIC:
            selected_indices = self._walk_ttbar_hadronic(gen_parts)
        elif self.sample_type == HHSampleType.TTBAR_SEMILEPTONIC:
            selected_indices = self._walk_ttbar_semileptonic(gen_parts)
        else:
            raise ValueError(f"Unknown sample type: {self.sample_type}")

        # Convert selected indices back to an awkward array mask
        selected_ak = ak.Array(selected_indices)
        columns[self.output_col] = gen_parts[selected_ak]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
