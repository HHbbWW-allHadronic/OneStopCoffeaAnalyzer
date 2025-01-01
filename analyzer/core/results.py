import logging
from typing import Any, Optional
from rich import print
import functools as ft
import operator as op

from collections import defaultdict
import pydantic as pyd
from analyzer.configuration import CONFIG
from analyzer.datasets import (
    SampleParams,
    FileSet,
    DatasetParams,
    SampleType,
    DatasetRepo,
)
import pickle as pkl
from analyzer.utils.structure_tools import accumulate
from .common_types import Scalar

import analyzer.datasets as ad
import analyzer.core.selection as ans
import analyzer.core.region_analyzer as anr
import analyzer.core.histograms as anh
import analyzer.core.specifiers as spec


if CONFIG.PRETTY_MODE:
    pass


logger = logging.getLogger(__name__)


class BaseResult(pyd.BaseModel):
    histograms: dict[str, anh.HistogramCollection] = pyd.Field(default_factory=dict)
    other_data: dict[str, Any] = pyd.Field(default_factory=dict)
    selection_flow: ans.SelectionFlow | None = None

    def __add__(self, other: "SubSectorResult"):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return BaseResult(
            histograms=new_hists,
            other_data=new_other,
            selection_flow=self.selection_flow + other.selection_flow,
        )

    def scaled(self, scale):
        return BaseResult(
            histograms={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            selection_flow=self.selection_flow,  # .scaled(scale),
        )


class SubSectorResult(pyd.BaseModel):
    region: anr.RegionAnalyzer
    base_result: BaseResult

    def __add__(self, other: "SubSectorResult"):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.region != other.region:
            raise RuntimeError("Different Regions")
        return SubSectorResult(
            region=self.region, base_result=self.base_result + other.base_result
        )

    def scaled(self, scale):
        return SubSectorResult(
            region=self.region, base_result=self.base_result.scaled(scale)
        )


class SampleResult(pyd.BaseModel):
    sample_id: ad.SampleId

    params: SampleParams
    results: dict[str, SubSectorResult]

    file_set_ran: FileSet
    file_set_processed: FileSet

    @property
    def processed_events(self):
        return self.file_set_processed.events

    def __add__(self, other):
        fs = self.file_set_processed.intersect(other.file_set_processed)
        if not fs.empty:
            error = (
                f"Could not add analysis results because the file sets over which they successfully processed overlap."
                f"Overlapping files:\n{fs}"
            )
            raise RuntimeError(error)

        if self.params != other.params or self.region != other.region:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )

        return SampleResult(
            sample_id=self.sample_id,
            params=self.params,
            file_set_ran=self.file_set_ran + other.file_set_ran,
            file_set_processed=self.file_set_processed + other.file_set_processed,
            results=accumulate([self.results, other.results]),
        )

    def scaled(self, scale):
        return SampleResult(
            sample_id=self.sample_id,
            params=self.params,
            file_set_ran=self.file_set_ran,
            file_set_processed=self.file_set_processed,
            results={k: v.scaled(scale) for k, v in self.results.items()},
        )

    def scaleToPhysical(self):
        if self.params.dataset.sample_type == SampleType.MC:
            scale = (
                self.params.dataset.lumi
                * self.params.x_sec
                / self.file_set_processed.events
            )
        else:
            scale = self.params.n_events / self.file_set_processed.events

        return self.scaled(scale)


class SectorResult(pyd.BaseModel):
    sector_params: spec.SectorParams
    result: BaseResult

    @property
    def params_dict(self):
        return self.sector_params.model_dump()


    @property
    def histograms(self):
        return self.result.histograms


class DatasetResult(pyd.BaseModel):
    dataset_params: DatasetParams
    results: dict[str, BaseResult]

    file_set_ran: FileSet
    file_set_processed: FileSet

    @staticmethod
    def fromSampleResult(sample_results):
        if not len(set(x.sample_id.dataset_name for x in sample_results)) == 1:
            raise RuntimeError()

        return DatasetResult(
            dataset_params=sample_results[0].params.dataset,
            results=accumulate(
                [
                    {k: v.base_result for k, v in r.results.items()}
                    for r in sample_results
                ]
            ),
            file_set_ran=accumulate([x.file_set_ran for x in sample_results]),
            file_set_processed=accumulate(
                [x.file_set_processed for x in sample_results]
            ),
        )

    def scaled(self, scale):
        return DatasetReult(
            dataset_params=self.dataset_params,
            region_results={k: v.scaled(scale) for k, v in results.items()},
            file_set_ran=self.file_set_ran,
            file_set_processed=self.file_set_processed,
        )

    def __add__(self, other):
        if self.dataset_params != other.dataset_params:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible results. The two results have different parameters."
            )
        return SectorResult(
            dataset_params=self.dataset_params,
            results=accumulate([self.result, other.results]),
            file_set_ran=self.file_set_ran + other.file_set_ran,
            file_set_processed=self.file_set_processed + other.file_set_processed,
        )

    @property
    def sector_results(self):
        return [
            SectorResult(
                sector_params=spec.SectorParams(
                    dataset=self.dataset_params, region_name=rn
                ),
                result=result,
            )
            for rn, result in self.results.items()
        ]


results_adapter = pyd.TypeAdapter(dict[ad.SampleId, SampleResult])
subsector_adapter = pyd.TypeAdapter(dict[str, SubSectorResult])


def loadResults(obj):
    return results_adapter.validate_python(obj)


def loadSampleResultFromPaths(paths):
    results = []
    for p in paths:
        with open(p, "rb") as f:
            data = pkl.load(f)
            r = loadResults(data)
            results.append(r)
    return accumulate(results)


def makeDatasetResults(sample_results):
    key = lambda x: x.sample_id.dataset_name
    grouped = it.groupby(sorted(sample_results.values, key=key), key=key)


def makeDatasetResults(sample_results, drop_samples=None):
    drop_samples = drop_samples or []
    scaled_sample_results = defaultdict(list)
    for result in sample_results.values():
        if result.sample_id in drop_samples:
            logger.warn(f'Not including sample "{sample_id}"')
            continue
        scaled_sample_results[result.sample_id.dataset_name].append(
            result.scaleToPhysical()
        )

    return {
        x: DatasetResult.fromSampleResult(y) for x, y in scaled_sample_results.items()
    }


def checkResult(paths):
    from rich.console import Console
    from rich.style import Style
    from rich.table import Table

    console = Console()

    results = list(loadSampleResultFromPaths(paths).values())

    table = Table(title="Missing Events")
    for x in ("Dataset Name", "Sample Name", "% Complete", "Processed", "Total"):
        table.add_column(x)

    for result in results:
        sample_id = result.params.sample_id
        exp = result.params.n_events
        val = result.processed_events
        diff = exp - val
        done = diff == 0
        percent = round(val / exp * 100, 2)

        table.add_row(
            sample_id.dataset_name,
            sample_id.sample_name,
            str(percent),
            f"{val}",
            f"{exp}",
            style=Style(color="green" if done else "red"),
        )
        # print(f"{sample_id} is missing {diff} events")
    console.print(table)
