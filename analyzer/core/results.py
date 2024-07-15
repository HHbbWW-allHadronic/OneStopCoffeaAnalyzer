import copy
import itertools as it
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
from analyzer.file_utils import stripPort

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
    processed_chunks: Any

    @property
    def raw_events_processed(self):
        return sum(e - s for _, s, e in self.processed_chunks)

    def getBadChunks(self):
        a = self.dataset_preprocessed.chunks
        b = self.processed_chunks
        return a.difference(b)

    def getMissingCoffeaDataset(self):
        missing_chunks = self.dataset_preprocessed.chunks.difference(
            self.getProcessedChunks()
        )
        input_dataset = copy.deepcopy(self.getCoffeaDataset())

        def filterSteps(name, steps, missing):
            return [s for s in steps if (name, *s) in missing]

        for k, v in input_dataset["files"].items():
            v["steps"] = filterSteps(k, v["steps"], missing_chunks)

        input_dataset["files"] = {
            k: v for k, v in input_dataset["files"].items() if v["steps"]
        }

        return input_dataset

    def getMissingDataset(self):
        ds_prepped = copy.deepcopy(self.dataset_preprocessed)
        ds_prepped.limit_chunks = self.getBadChunks()
        return ds_prepped

    def getScaledHistograms(self, sample_manager, target_lumi):
        sample = sample_manager[self.dataset_preprocessed.dataset_input.dataset_name]
        weight = sample.getWeight(target_lumi)
        reweighted = sample.n_events / self.raw_events_processed
        final_weight = reweighted * weight
        return {name: h * final_weight for name, h in self.histograms.items()}

    def merge(self, other):
        if (
            self.dataset_preprocessed.dataset_name
            != other.dataset_preprocessed.dataset_name
        ):
            raise ValueError()
        if self.processed_chunks.intersection(other.processed_chunks):
            raise ValueError()
        new_hists = utils.accumulate([self.histograms, other.histograms])
        result = DatasetRunResult(
            self.dataset_preprocessed,
            new_hists,
            self.processed_chunks | other.processed_chunks,
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
    module_list: List[str]

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
        return utils.accumulate(
            [
                {
                    v.dataset_preprocessed.dataset_input.fill_name: v.getScaledHistograms(
                        sample_manager, target_lumi
                    )
                }
                for k, v in self.results.items()
            ]
        )
        # return mergeAndWeightResults(self.results.values(), sample_manager, target_lumi)

    def merge(self, other):
        updated_results = copy.deepcopy(self.results)
        for dataset_name, results in other.results.items():
            if dataset_name in updated_results:
                updated_results[dataset_name] = updated_results[dataset_name].merge(
                    results
                )
            else:
                updated_results[dataset_name] = results
        return AnalysisResult(updated_results, self.module_list)


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
            bad_chunks = result.getBadChunks()
            diff = expected - actual
            percent = diff / expected * 100
            return AnalysisInspectionResult(
                "Number Events",
                False,
                f"Expected {expected}, found {actual}. Missing {diff} ({percent:0.2f}%) in {len(bad_chunks)} bad chunks.",
            )


class InputChecker:
    def __init__(self, sample_manager):
        self.sample_manager = sample_manager

    def __call__(self, result):
        missing_files = result.dataset_preprocessed.missingFiles()
        if missing_files:
            return AnalysisInspectionResult(
                "Preprocessing",
                False,
                f"Missing {len(missing_files)} file{'s' if len(missing_files) > 1 else ''} from preprocessed input. This means not all files were preprocessed correctly.",
            )
        else:
            return AnalysisInspectionResult(
                "Preprocessing",
                True,
                f"All files peprocessed",
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
