import logging
import pickle as pkl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from urllib import parse

import analyzer.utils as utils
import awkward as ak
import dask_awkward as dak
import hist
import hist.dask as dah

from .inputs import DatasetPreprocessed

logger = logging.getLogger()


@dataclass
class ResultModification:
    user: str
    time: datetime


@dataclass
class DatasetDaskRunResult:
    dataset_preprocessed: DatasetPreprocessed
    histograms: Dict[str, dah.Hist]
    run_report: dak.Array


    def getName(self):
        return self.dataset_preprocessed.dataset_input.dataset_name


@dataclass
class DatasetRunResult:
    dataset_preprocessed: DatasetPreprocessed
    histograms: Dict[str, hist.Hist]
    dataset_run_report: ak.Array

    @property
    def raw_events_processed(self):
        good_mask = ak.is_none(self.dataset_run_report["exception"])
        rr = self.dataset_run_report[good_mask]
        starts = ak.strings_astype(rr["args"][:,2],int)
        ends = ak.strings_astype(rr["args"][:,3],int)
        return ak.sum(ends-starts)

    def getBadChunks(self):
        good_mask = ak.is_none(self.dataset_run_report["exception"])
        rr = self.dataset_run_report[~good_mask]
        return rr
        

    def getScaledHistograms(self, sample_manager, target_lumi):
        sample = sample_manager[self.dataset_preprocessed.dataset_input.dataset_name]
        weight = sample.getWeight(target_lumi)
        reweighted = sample.n_events / self.raw_events_processed
        final_weight = reweighted * weight
        return {name: h * final_weight for name, h in self.histograms.items()}

    def update(self, other):
        if self.dataset_preprocessed != other.dataset_preprocessed:
            raise ValueError()
        new_hists = accumulate([self.histograms, other.histograms])
        total_events = self.raw_events_processed + other.raw_events_processed
        report = ak.concat(self.dataset_run_report, other.dataset_run_report)
        result = DatasetRunResult(
            self.dataset_preprocessed, new_hists, total_events, report
        )
        return result

    def getName(self):
        return self.dataset_preprocessed.dataset_input.dataset_name


def mergeAndWeightResults(results, sample_manager, target_lumi=None):
    return utils.accumulate(
        [x.getScaledHistograms(sample_manager, target_lumi) for x in results]
    )


@dataclass
class AnalysisResult:
    # modifications: List[ResultModification]
    results: Dict[str, DatasetRunResult]

    def save(self, output_file):
        path = Path(output_file)
        parent = path.parent
        parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def fromFile(path):
        path = Path(path)
        with open(path, "rb") as f:
            ret = pkl.load(f)
        if not isinstance(ret, AnalysisResult):
            raise RuntimeError(f"File {path} does not contain an analysis result")
        return ret

    def getMergedHistograms(self, sample_manager, target_lumi=None):
        return mergeAndWeightResults(self.results.values(), sample_manager, target_lumi)


@dataclass
class AnalysisInspectionResult:
    type: str
    passed: bool
    description: str


class NEventChecker:
    def __init__(self, sample_manager):
        self.sample_manager = sample_manager

    def __call__(self, result):
        expected = self.sample_manager.getSet(result.getName()).n_events
        actual = result.raw_events_processed
        if expected == actual:
            return AnalysisInspectionResult(
                "Number Events",
                True,
                f"Expected {expected}, found {actual}",
            )
        else:
            return AnalysisInspectionResult(
                "Number Events",
                False,
                f"Expected {expected}, found {actual}",
            )


class InputChecker:
    def __init__(self, sample_manager):
        self.sample_manager = sample_manager

    def __call__(self, result):
        sample = self.sample_manager.getSet(result.getName())
        files = [tuple(parse.urlparse(y)[2] for y in x.paths.values()) for x in sample.files]
        prepped = result.dataset_preprocessed
        cof_dataset = prepped.coffea_dataset_split
        cof_files = set(parse.urlparse(x)[2] for x in cof_dataset["files"].keys())
        diff = [x for x in files if not any(y in cof_files for y in x)]
        if diff:
            return AnalysisInspectionResult(
                "Input Files",
                False,
                f"Missing {len(diff)} file{'s' if len(diff) > 1 else ''} from analysis input",
            )
        else:
            return AnalysisInspectionResult(
                "Input Files",
                True,
                f"All files in sample found in input to analyzer",
            )


def checkDatasetResult(ds_result, sample_manager):
    checkers = [NEventChecker(sample_manager), InputChecker(sample_manager)]
    results = [checker(ds_result) for checker in checkers]
    return results


def checkAnalysisResult(result, sample_manager):
    ret = {
        name: checkDatasetResult(ds_res, sample_manager)
        for name, ds_res in result.results.items()
    }
    return ret
