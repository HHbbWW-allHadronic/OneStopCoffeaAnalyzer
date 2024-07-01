import logging
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
import sys

import time
import awkward as ak
import dask
import dask.base as daskb
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from distributed import Client, get_client, rejoin, secede

import dask.optimization as dop

from .events import getEvents
from .inputs import DatasetPreprocessed
from .lumi import getLumiMask
from .org import AnalyzerModule, generateTopology, namesToModules, sortModules
from .processor import DatasetProcessor
from .results import DatasetDaskRunResult, DatasetRunResult

logger = logging.getLogger(__name__)

def execute(futures: Iterable[DatasetDaskRunResult], client: Client):
    futures = list(futures)
    logger.debug(f"Executing {len(futures)} analysis futures.")
    dsk = {
        x.getName(): [
            x.dataset_preprocessed,
            x.histograms,
            x.run_report,
        ]
        for x in futures
    }

    computed, *rest = dask.compute(dsk)

    return {
        name: DatasetRunResult(prep, h, rep)
        for name, (prep, h,  rep) in computed.items()
    }

# def _execute(futures: Iterable[DatasetDaskRunResult], client: Client):
#     futures = list(futures)
#     logger.debug(f"Executing {len(futures)} analysis futures.")
#     dsk = {
#         x.getName(): [
#             x.dataset_preprocessed,
#             x.histograms,
#             x.raw_events_processed,
#             x.run_report,
#         ]
#         for x in futures
#     }
#     computed, *rest = compute(dsk, optimize_graph=False)
# 
#     return {
#         name: DatasetRunResult(prep, h, r, rep)
#         for name, (prep, h, r, rep) in computed.items()
#     }


class Analyzer:
    """
    Represents an analysis, a collection of modules.
    """

    def __init__(self, modules: Iterable[AnalyzerModule], cache: Any):
        self.modules: List[AnalyzerModule] = self.__createAndSortModules(*modules)
        logger.info(
            "Will run modules in the following order:\n"
            + "\n".join(f"\t{i+1}. {x.name}" for i, x in enumerate(self.modules))
        )
        self.cache = cache

    def __createAndSortModules(self, *module_names):
        m = namesToModules(module_names)
        t = generateTopology(m)
        modules = sortModules(m)
        return modules

    def getDatasetFutures(
        self, dsprep: DatasetPreprocessed, delayed: bool = True
    ) -> DatasetDaskRunResult:
        dataset_name = dsprep.dataset_input.dataset_name
        lumi_json = dsprep.dataset_input.lumi_json
        logger.debug(f"Generating futures for dataset {dataset_name}")
        files = dsprep.getCoffeaDataset()["files"]
        files = {k:v for k,v in files.items() if v["num_entries"] is not None}
        maybe_base_form = dsprep.coffea_dataset_split.get("form", None)
        if maybe_base_form is not None:
            maybe_base_form = ak.forms.from_json(decompress_form(maybe_base_form))
        events, report = getEvents(files, maybe_base_form, self.cache)
        
        if lumi_json:
            logger.info(f'Dataset {dataset_name}: Using lumi json file "{lumi_json}".')
            lmask = getLumiMask(lumi_json)
            events = events[lmask(events.run, events.luminosityBlock)]

        if delayed:
            daskres = DatasetDaskRunResult(dsprep, {}, report)
        else:
            events = events.compute(scheduler="synchronous")
            report = report.compute(scheduler="single-threaded")
            daskres = DatasetRunResult(dsprep, {}, report)

        dataset_analyzer = DatasetProcessor(
            daskres,
            dsprep.dataset_input.fill_name,
            dsprep.dataset_input.profile,
            delayed=delayed,
        )
        for m in self.modules:
            logger.info(f"Adding module {m.name} to dataset {dataset_name}")
            test = m(events, dataset_analyzer)
            events, dataset_analyzer = test

        return daskres
