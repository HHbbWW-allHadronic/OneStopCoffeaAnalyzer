from analyzer.core.analysis_modules import AnalyzerModule
import re
import numba
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
        dr_all = qi.delta_r(qj)
        min_dr = ak.min(dr_all, axis=1)
        # Replace None (from empty events) with -1 so histogram can handle it
        columns[self.output_col] = ak.fill_none(min_dr, -1.0)
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

class HHSampleType(enum.Enum):
    SIGNAL = "signal"
    TTBAR_HADRONIC = "ttbar_hadronic"
    TTBAR_SEMILEPTONIC = "ttbar_semileptonic"

@numba.njit
def _walk_signal_event(ev_mother, ev_pid, ev_status, ev_candidates):
    n = len(ev_pid)
    selected = np.zeros(n, dtype=np.int64)
    count = 0
    visited = np.zeros(n, dtype=numba.boolean)

    for i in range(n):
        if not ev_candidates[i]:
            continue

        visited[:] = False
        current = int(ev_mother[i])
        found_higgs = False

        while current >= 0 and current < n and not visited[current]:
            visited[current] = True
            if abs(ev_pid[current]) == 25 and (ev_status[current] >> 13) & 1 == 1:
                found_higgs = True
                break
            current = int(ev_mother[current])

        if found_higgs:
            selected[count] = i
            count += 1

    return selected[:count]


@numba.njit
def _walk_ttbar_hadronic_event(ev_mother, ev_pid, ev_status, ev_candidates):
    n = len(ev_pid)
    selected = np.zeros(n, dtype=np.int64)
    count = 0
    visited = np.zeros(n, dtype=numba.boolean)

    for i in range(n):
        if not ev_candidates[i]:
            continue

        visited[:] = False
        current = int(ev_mother[i])
        found_top = False

        while current >= 0 and current < n and not visited[current]:
            visited[current] = True
            if abs(ev_pid[current]) == 6:
                found_top = True
                break
            current = int(ev_mother[current])

        if found_top:
            selected[count] = i
            count += 1

    return selected[:count]


@numba.njit
def _walk_ttbar_semileptonic_event(ev_mother, ev_pid, ev_status, ev_candidates):
    n = len(ev_pid)
    selected = np.zeros(n, dtype=np.int64)
    count = 0
    visited = np.zeros(n, dtype=numba.boolean)

    for i in range(n):
        if not ev_candidates[i]:
            continue

        pid = abs(ev_pid[i])

        # b quarks — need top ancestry
        if pid == 5:
            visited[:] = False
            current = int(ev_mother[i])
            found_top = False

            while current >= 0 and current < n and not visited[current]:
                visited[current] = True
                if abs(ev_pid[current]) == 6:
                    found_top = True
                    break
                current = int(ev_mother[current])

            if found_top:
                selected[count] = i
                count += 1

        # light quarks — need hadronic W ancestry
        elif pid == 1 or pid == 2 or pid == 3 or pid == 4:
            visited[:] = False
            current = int(ev_mother[i])
            found_hadronic_w = False

            while current >= 0 and current < n and not visited[current]:
                visited[current] = True
                if abs(ev_pid[current]) == 24:
                    w_mother = int(ev_mother[current])
                    if w_mother >= 0 and abs(ev_pid[w_mother]) == 6:
                        found_hadronic_w = True
                    break
                current = int(ev_mother[current])

            if found_hadronic_w:
                selected[count] = i
                count += 1

    return selected[:count]


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

    def _is_bit_set(self, value, bit):
        return (value >> bit) & 1 == 1

    def _walk_signal(self, gen_parts):
        result_mask_events = []

        abs_pid = abs(gen_parts.pdgId)
        is_first_copy = (gen_parts.statusFlags >> 12) & 1 == 1
        is_hard_process = (gen_parts.statusFlags >> 7) & 1 == 1
        is_quark = (abs_pid >= 1) & (abs_pid <= 5)
        candidates = is_first_copy & is_hard_process & is_quark

        for ev in range(len(gen_parts)):
            ev_mother = ak.to_numpy(gen_parts[ev].genPartIdxMother).astype(np.int64)
            ev_pid = ak.to_numpy(gen_parts[ev].pdgId).astype(np.int64)
            ev_status = ak.to_numpy(gen_parts[ev].statusFlags).astype(np.int64)
            ev_candidates = ak.to_numpy(candidates[ev])

            selected = _walk_signal_event(ev_mother, ev_pid, ev_status, ev_candidates)

            if len(selected) != 6:
                #logger.warning(
                #    f"Signal event {ev}: expected 6 quarks, found {len(selected)}, "
                #    f"skipping (likely semi-leptonic decay)."
                #)
                result_mask_events.append(np.array([], dtype=np.int64))
                continue

            result_mask_events.append(selected)
        return result_mask_events

    def _walk_ttbar_hadronic(self, gen_parts):
        result_mask_events = []

        abs_pid = abs(gen_parts.pdgId)
        is_first_copy = (gen_parts.statusFlags >> 12) & 1 == 1
        is_hard_process = (gen_parts.statusFlags >> 7) & 1 == 1
        is_quark = (abs_pid >= 1) & (abs_pid <= 5)
        candidates = is_first_copy & is_hard_process & is_quark

        for ev in range(len(gen_parts)):
            ev_mother = ak.to_numpy(gen_parts[ev].genPartIdxMother).astype(np.int64)
            ev_pid = ak.to_numpy(gen_parts[ev].pdgId).astype(np.int64)
            ev_status = ak.to_numpy(gen_parts[ev].statusFlags).astype(np.int64)
            ev_candidates = ak.to_numpy(candidates[ev])

            selected = _walk_ttbar_hadronic_event(ev_mother, ev_pid, ev_status, ev_candidates)

            if len(selected) != 6:
                logger.warning(f"TTbar hadronic event {ev}: expected 6 quarks, found {len(selected)}.")
                for i in selected:
                    mother_pid = int(ev_pid[ev_mother[i]]) if ev_mother[i] >= 0 else None
                    logger.warning(
                        f"  idx={i}, pid={int(ev_pid[i])}, "
                        f"statusFlags={int(ev_status[i])}, "
                        f"mother_idx={int(ev_mother[i])}, "
                        f"mother_pid={mother_pid}"
                    )
                raise ValueError(
                    f"TTbar hadronic event {ev}: expected 6 quarks, found {len(selected)}. "
                    f"Check decay chain integrity."
                )
            result_mask_events.append(selected)
        return result_mask_events

    def _walk_ttbar_semileptonic(self, gen_parts):
        result_mask_events = []

        abs_pid = abs(gen_parts.pdgId)
        is_first_copy = (gen_parts.statusFlags >> 12) & 1 == 1
        is_hard_process = (gen_parts.statusFlags >> 7) & 1 == 1
        is_quark = (abs_pid >= 1) & (abs_pid <= 5)
        candidates = is_first_copy & is_hard_process & is_quark

        for ev in range(len(gen_parts)):
            ev_mother = ak.to_numpy(gen_parts[ev].genPartIdxMother).astype(np.int64)
            ev_pid = ak.to_numpy(gen_parts[ev].pdgId).astype(np.int64)
            ev_status = ak.to_numpy(gen_parts[ev].statusFlags).astype(np.int64)
            ev_candidates = ak.to_numpy(candidates[ev])

            selected = _walk_ttbar_semileptonic_event(ev_mother, ev_pid, ev_status, ev_candidates)

            if len(selected) != 4:
                logger.warning(f"TTbar semileptonic event {ev}: expected 4 quarks, found {len(selected)}.")
                for i in selected:
                    mother_pid = int(ev_pid[ev_mother[i]]) if ev_mother[i] >= 0 else None
                    logger.warning(
                        f"  idx={i}, pid={int(ev_pid[i])}, "
                        f"statusFlags={int(ev_status[i])}, "
                        f"mother_idx={int(ev_mother[i])}, "
                        f"mother_pid={mother_pid}"
                    )
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

        selected_lists = [arr.tolist() for arr in selected_indices]
        selected_ak = ak.Array(selected_lists)

        is_fully_hadronic = ak.num(selected_ak, axis=1) == 6
        
        columns[self.output_col] = gen_parts[selected_ak]
        addSelection(columns, "is_fully_hadronic", is_fully_hadronic)
        
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [
            self.output_col,
            Column(("Selection", "is_fully_hadronic")),
    ]
