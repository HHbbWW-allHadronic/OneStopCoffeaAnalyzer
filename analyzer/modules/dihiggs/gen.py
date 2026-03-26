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

@define
class GenPartMaxDRMaker(AnalyzerModule):
    """
    Computes two max delta R variables for a collection of 4 light quarks:
    1) max_dr_4q: maximum delta R among all 6 unique pairs of the 4 quarks,
       no conditions applied.
    2) max_dr_3q: for events where max_dr_4q > isolation_threshold, remove
       the quark that contributes most to the max dR (most isolated from the
       other three), then compute max dR of remaining 3 quarks.
       Filled with -1.0 for events where max_dr_4q <= isolation_threshold.

    Parameters
    ----------
    input_col : Column
        Column containing the 4 light quarks (GenPart_4q).
    output_col_4q : Column
        Column where max delta R of all 4 quarks will be stored.
    output_col_3q : Column
        Column where max delta R after removing most isolated quark will be stored.
    isolation_threshold : float, optional
        Delta R threshold above which we remove the most isolated quark.
        By default 0.8 (AK8 fat jet radius).
    """
    input_col: Column
    output_col_4q: Column
    output_col_3q: Column
    isolation_threshold: float = 0.8

    def run(self, columns, params):
        quarks = columns[self.input_col]

        # Compute all pairwise dR using combinations
        pairs = ak.combinations(quarks, 2, axis=1)
        q_a, q_b = ak.unzip(pairs)
        dr_pairs = q_a.delta_r(q_b)

        # Max dR among all 6 pairs — no conditions
        max_dr_4q = ak.fill_none(ak.max(dr_pairs, axis=1), -1.0)
        columns[self.output_col_4q] = max_dr_4q

        # For each quark find its min dR to any other quark
        dr_matrix = quarks[:, :, np.newaxis].delta_r(quarks[:, np.newaxis, :])
        large_val = 999.0
        local_idx = ak.local_index(quarks, axis=1)
        dr_no_self = ak.where(
            local_idx[:, :, np.newaxis] == local_idx[:, np.newaxis, :],
            large_val,
            dr_matrix
        )
        min_dr_to_others = ak.min(dr_no_self, axis=2)

        # Remove most isolated quark and compute max dR of remaining 3 — all events
        isolated_idx = ak.argmax(min_dr_to_others, axis=1, keepdims=True)
        keep_mask = local_idx != isolated_idx
        remaining_quarks = quarks[keep_mask]
        pairs_3q = ak.combinations(remaining_quarks, 2, axis=1)
        q_a3, q_b3 = ak.unzip(pairs_3q)
        dr_pairs_3q = q_a3.delta_r(q_b3)
        max_dr_3q = ak.fill_none(ak.max(dr_pairs_3q, axis=1), -1.0)
        columns[self.output_col_3q] = max_dr_3q

        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col_4q, self.output_col_3q]
