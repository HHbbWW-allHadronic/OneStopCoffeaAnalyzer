# from __future__ import annotations
import concurrent.futures
import copy
import functools as ft
import itertools as it
import json
import logging
import operator as op
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import awkward as ak
import dask
import pydantic as pyd
import yaml
from analyzer.configuration import CONFIG
from analyzer.datasets import (
    DatasetRepo,
    EraRepo,
    SampleId,
    SampleType,
    SampleParams,
    FileSet,
)
from analyzer.utils.file_tools import extractCmsLocation
from analyzer.utils.structure_tools import accumulate
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from rich import print
from .common_types import Scalar

import analyzer.core.branch_analyzer as ana
import analyzer.core.selection as ans
import analyzer.core.histograms as anh
import analyzer.core.specifiers as anp

if CONFIG.PRETTY_MODE:
    from rich.progress import track


logger = logging.getLogger(__name__)


class SelectionResult(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    cutflow: ans.SelectionFlow
    weighted_sum: Optional[tuple[Scalar, Scalar]]

    def __add__(self, other):
        """Two cutflows may be sumed by simply adding them"""
        if self.weighted_sum is not None:
            new_ws = (
                (self.weighted_sum[0] + other.weighted_sum[0]),
                (self.weighted_sum[1] + other.weighted_sum[1]),
            )
        else:
            new_ws = None

        return SelectionResult(
            cutflow=self.cutflow + other.cutflow,
            weighted_sum=new_ws,
        )

    # def scaled(self, scale):
    #     if self.weighted_sum:
    #         nws = (
    #             self.weighted_sum[0] * scale,
    #             self.weighted_sum[1] * (scale**2),
    #         )
    #     else:
    #         nws = None
    #     return SelectionResult(
    #         cutflow=self.cutflow,
    #         weighted_sum=nws,
    #     )


class CoreSubSectorResult(pyd.BaseModel):
    region: ana.RegionAnalyzer
    params: SampleParams
    histograms: dict[str, anh.HistogramCollection] = pyd.Field(default_factory=dict)
    other_data: dict[str, Any] = pyd.Field(default_factory=dict)
    selection_flow: Optional[ans.SelectionFlow] = None

    def __add__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.params != other.params or self.region != other.region:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return SubSectorResult(
            file_set=self.file_set + other.file_set,
            region=self.region,
            params=self.params,
            histograms=new_hists,
            other_data=new_other,
            selection_flow=self.selection_flow + other.selection_flow,
        )

    def scaled(self, scale):
        return SubSectorResult(
            file_set=self.file_set,
            region=self.region,
            params=self.params,
            histograms={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            selection_flow=self.selection_flow,  # .scaled(scale),
        )


class SubSectorResult(pyd.BaseModel):
    file_set_target: FileSet
    file_set_processed: FileSet
    core_result: CoreSubSectorResult

    def __add__(self, other):
        return CoreSubSectorResult(
            file_set_target=self.file_set_target,
            file_set_processed=self.file_set_processed + other.file_set_processed,
            core_result=self.core_result + other.core_result,
        )

    def scaled(self, scale):
        return SubSectorResult(
            file_set_target=self.file_set_target,
            file_set_processed=self.file_set_processed + other.file_set_processed,
            core_result=self.core_result.scaled(scale)
        )


class SectorResult(pyd.BaseModel):
    sector_params: anp.SectorParams
    sample_params: list[dict[str, Any]]
    histograms: dict[str, anh.HistogramCollection]
    other_data: dict[str, Any]
    cutflow_data: Optional[ans.SelectionFlow] = None

    @staticmethod
    def fromSubSectorResult(subsector_result):
        return SectorResult(
            sector_params=subsector_result.params.sector,
            sample_params=[subsector_result.params.sample],
            histograms=subsector_result.histograms,
            other_data=subsector_result.other_data,
            cutflow_data=subsector_result.cutflow_data,
        )

    def scaled(self, scale):
        return SectorResult(
            sector_params=self.sector_params,
            sample_params=[subsector_result.params.sample_params],
            histogrmams={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            cutflow_data=self.cutflow_data.scaled(scale),
        )

    def __add__(self, other):
        """Two subsector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.sector_params != other.sector_params:
            raise RuntimeError(f"Error: Attempting to merge incomaptible results")
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return SectorResult(
            sector_params=self.sector_params,
            sample_params=self.sample_params + other.sample_params,
            histograms=new_hists,
            other_data=new_other,
            cutflow_data=self.cutflow_data + other.cutflow_data,
        )


# class AnalysisResult(pyd.BaseModel):
#     model_config = pyd.ConfigDict(arbitrary_types_allowed=True)
#     description: AnalysisDescription
#     preprocessed_samples: dict[anp.SampleId, SamplePreprocessed]
#     processed_chunks: dict[SampleId, Any]
#     results: dict[SubSectorId, SubSectorResult]
#     total_mc_weights: dict[SampleId, Optional[Scalar]]
#
#     @property
#     def raw_events_processed(self):
#         return {
#             sample_id: sum(e - s for _, s, e in chunks)
#             for sample_id, chunks in self.processed_chunks.items()
#         }
#
#     def getBadChunks(self):
#         ret = {
#             n: self.preprocessed_samples[n].chunks.difference(
#                 self.processed_chunks.get(n, set())
#             )
#             for n in self.preprocessed_samples
#         }
#         return ret
#
#     def isEmpty(self):
#         return self.limit_chunks is not None and not self.limit_chunks
#
#     def createMissingRunnable(self):
#         logger.debug(f"Scanning for bad chunks")
#         prepped = copy.deepcopy(self.preprocessed_samples)
#         bad_chunks = self.getBadChunks()
#         ret = {}
#         for sid in prepped:
#             logger.info(f"Sample {sid} has {len(bad_chunks[sid])} bad chunks.")
#             bad = bad_chunks[sid]
#             if bad:
#                 ret[sid] = prepped[sid]
#                 ret[sid].limit_chunks = bad
#         print(list(ret.keys()))
#         return ret
#
#     def __add__(self,eights[k] = v
#
#         new_weights = accumulate([self.total_mc_weights, other.total_mc_weights])
#         return AnalysisResult(
#             description=self.description,
#             preprocessed_samples=self.preprocessed_samples,
#             processed_chunks=self.processed_chunks,
#             results=new_results,
#             total_mc_weights=new_weights,
#
#         )
#
#     def getResults(self, drop_samples=None):
#         drop_samples = drop_samples or []
#         scaled_sample_results = defaultdict(list)
#         for subsector_id, result in self.results.items():
#             if subsector_id.sample_id in drop_samples:
#                 logger.warn(f"Not including subsector \"{subsector_id}\"")
#                 continue
#             k = (subsector_id.sample_id.dataset_name, subsector_id.region_name)
#
#             if result.params.sector.dataset.sample_type == SampleType.MC:
#                 sample_info = result.params.sample
#                 scale = (
#                     result.params.sector.dataset.lumi
#                     * sample_info["x_sec"]
#                     / self.total_mc_weights[subsector_id.sample_id]
#                 )
#                 result = result.scaled(scale)
#             scaled_sample_results[k].append(SectorResult.fromSubSectorResult(result))
#
#         return {x: ft.reduce(op.add, y) for x, y in scaled_sample_results.items()}
#
#     @staticmethod
#     def fromFile(path):
#         with open(path, "rb") as f:
#             data = pkl.load(f)
#         return AnalysisResult(**data)
#
#
# def checkResult(input_path):
#     from rich.console import Console
#     from rich.style import Style
#     from rich.table import Table
#
#     console = Console()
#
#     with open(input_path, "rb") as f:
#         result = pkl.load(f)
#         result = AnalysisResult(**result)
#     dr = DatasetRepo.getConfig()
#     wanted_samples = sorted(list(result.preprocessed_samples))
#     processed = result.raw_events_processed
#
#     table = Table(title="Missing Events")
#     for x in ("Dataset Name", "Sample Name", "% Complete", "Processed", "Total"):
#         table.add_column(x)
#
#     for sample_id in wanted_samples:
#         exp = dr.getSample(sample_id).n_events
#         val = processed.get(sample_id, 0)
#         diff = exp - val
#         done = diff == 0
#         percent = round(val / exp * 100, 2)
#
#         table.add_row(
#             sample_id.dataset_name,
#             sample_id.sample_name,
#             str(percent),
#             f"{val}",
#             f"{exp}",
#             style=Style(color="green" if done else "red"),
#         )
#         # print(f"{sample_id} is missing {diff} events")
#     console.print(table)
