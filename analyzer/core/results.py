# from __future__ import annotations
import copy
import functools as ft
import operator as op
import pickle as pkl
from collections import defaultdict
from typing import Any, Optional


import pydantic as pyd
from analyzer.configuration import CONFIG
from analyzer.datasets import SampleId, SampleType
from analyzer.utils.structure_tools import accumulate

from .common_types import Scalar
from .configuration import AnalysisDescription
from .histograms import HistogramCollection
from .preprocessed import SamplePreprocessed
from .selection import Cutflow
from .specifiers import SectorParams, SubSectorId, SubSectorParams

if CONFIG.PRETTY_MODE:
    pass

class SelectionResult(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    cutflow: Cutflow
    raw_passed: Scalar
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
            raw_passed=self.raw_passed + other.raw_passed,
            weighted_sum=new_ws,
        )

    def scaled(self, scale):
        if self.weighted_sum:
            nws = (
                self.weighted_sum[0] * scale,
                self.weighted_sum[1] * (scale**2),
            )
        else:
            nws = None
        return SelectionResult(
            cutflow=self.cutflow,
            raw_passed=self.raw_passed,
            weighted_sum=nws,
        )


class SubSectorResult(pyd.BaseModel):
    params: SubSectorParams
    histograms: dict[str, HistogramCollection] = pyd.Field(default_factory=dict)
    other_data: dict[str, Any] = pyd.Field(default_factory=dict)
    cutflow_data: Optional[SelectionResult] = None

    def __add__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.params != other.params:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return SubSectorResult(
            params=self.params,
            histograms=new_hists,
            other_data=new_other,
            cutflow_data=self.cutflow_data + other.cutflow_data,
        )

    def scaled(self, scale):
        return SubSectorResult(
            params=self.params,
            histograms={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            cutflow_data=self.cutflow_data.scaled(scale),
        )


class SectorResult(pyd.BaseModel):
    sector_params: SectorParams
    sample_params: list[dict[str, Any]]
    histograms: dict[str, HistogramCollection]
    other_data: dict[str, Any]
    cutflow_data: Optional[SelectionResult]

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


class AnalysisResult(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)
    description: AnalysisDescription
    preprocessed_samples: dict[SampleId, SamplePreprocessed]
    processed_chunks: dict[SampleId, Any]
    results: dict[SubSectorId, SubSectorResult]
    total_mc_weights: dict[SampleId, Optional[Scalar]]

    @property
    def raw_events_processed(self):
        return {
            sample_id: sum(e - s for _, s, e in chunks)
            for sample_id, chunks in self.processed_chunks.items()
        }

    def getBadChunks(self):
        ret = {
            n: self.preprocessed_samples[n].chunks.difference(self.processed_chunks[n])
            for n in self.preprocessed_samples
        }
        return ret

    def getMissingPreprocessed(self):
        logger.debug(f"Scanning for bad chunks")
        prepped = copy.deepcopy(self.preprocessed_samples)
        bad_chunks = self.getBadChunks()
        ret = {}
        for sid in prepped:
            logger.debug(f"Sample {sid} has {len(bad_chunks[sid])} bad chunks.")
            bad = bad_chunks[sid]
            if bad:
                ret[sid] = prepped[sid]
                ret[sid].limit_chunks = bad
        return ret

    def __add__(self, other):
        """Add two analysis results together.
        Two results may be added if they come from the same configuration, and do not have any overlapping
        chunks.
        """

        if self.description != other.description:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )

        if any(
            self.processed_chunks.get(x, set()).intersect(
                other.processed_chunk.get(x, set())
            )
            for x in set(self.processed_chunks) | set(other.processed_chunks)
        ):
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )

        desc = copy.copy(self.description)
        new_results = accumulate([self.results, other.results])
        return AnalysisResult(
            description=self.description,
            preprocessed_samples=self.preprocessed_samples,
            processed_chunks=self.processed_chunks,
            results=new_results,
        )

    def getResults(self, drop_samples=None):
        drop_samples = drop_samples or []
        scaled_sample_results = defaultdict(list)
        for subsector_id, result in self.results.items():
            if subsector_id.sample_id in drop_samples:
                continue
            k = (subsector_id.sample_id.dataset_name, subsector_id.region_name)

            if result.params.sector.dataset.sample_type == SampleType.MC:
                sample_info = result.params.sample
                scale = (
                    result.params.sector.dataset.lumi
                    * sample_info["x_sec"]
                    / self.total_mc_weights[subsector_id.sample_id]
                )
                result = result.scaled(scale)
            scaled_sample_results[k].append(SectorResult.fromSubSectorResult(result))

        return {x: ft.reduce(op.add, y) for x, y in scaled_sample_results.items()}

    @staticmethod
    def fromFile(path):
        with open(path, "rb") as f:
            data = pkl.load(f)
        return AnalysisResult(**data)
