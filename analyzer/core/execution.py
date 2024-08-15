import logging
import uproot
import sys
import time
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

import awkward as ak
import dask
import dask.base as daskb
import dask.optimization as dop
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from distributed import Client, get_client, rejoin, secede, progress

import dask.optimization as dop

from .events import getEvents
from .inputs import DatasetPreprocessed
from .org import AnalyzerModule, generateTopology, namesToModules, sortModules
from .processor import DatasetProcessor
from .results import DatasetDaskRunResult, DatasetRunResult
from analyzer.file_utils import stripPort, extractCmsLocation

logger = logging.getLogger(__name__)


def getProcessedChunks(run_report):
    good_mask = ak.is_none(run_report["exception"])
    rr = run_report[good_mask]
    return {
        (
            extractCmsLocation(x["args"][0][1:-1]),
            int(x["args"][2]),
            int(x["args"][3]),
        )
        for x in rr
    }


def execute(futures: Iterable[Tuple[DatasetDaskRunResult, Any]], client: Client):
    futures = list(futures)
    logger.debug(f"Executing {len(futures)} analysis futures.")
    dsk = {
        "main": {
            x.getName(): [
                x.dataset_preprocessed,
                x.histograms,
                x.non_scaled_histograms,
                x.non_scaled_histograms_labels,
                x.run_report,
                x.cut_list,
            ]
            for x, _ in futures
        },
        "side_effects": [y for _, y in futures],
    }
    computed, *rest = dask.compute(dsk, retries=3)

    ret = {
        name: DatasetRunResult(prep, h, getProcessedChunks(rep), nsh, nshl, cuts)
        for name, (prep, h, nsh, nshl, rep, cuts) in computed["main"].items()
    }

    return ret


class Analyzer:
    def __init__(self, modules: Iterable[AnalyzerModule], cache: Any):
        self.module_names = modules
        self.cache = cache

    def __createAndSortModules(self, sample_info, module_names, include_defaults=True):
        m = namesToModules(module_names)
        modules = sortModules(m, sample_info, include_defaults)
        return modules

    def getDatasetFutures(
        self,
        dsprep: DatasetPreprocessed,
        delayed: bool = True,
        skim_save_path: str = None,
        prog_bar_updater=None,
        file_retrieval_kwargs=None,
        include_default_modules=True,
    ) -> DatasetDaskRunResult:
        this_dataset_modules = self.__createAndSortModules(
            dsprep.dataset_input.sample_info,
            self.module_names,
            include_defaults=include_default_modules,
        )
        logger.info(
            "Will run modules in the following order:\n"
            + "\n".join(
                f"\t{i+1}. {x.name}" for i, x in enumerate(this_dataset_modules)
            )
        )
        if file_retrieval_kwargs is None:
            file_retrieval_kwargs = {}

        dataset_name = dsprep.dataset_input.dataset_name
        logger.debug(f"Generating futures for dataset {dataset_name}.")
        files = dsprep.getCoffeaDataset(**file_retrieval_kwargs)["files"]
        files = {k: v for k, v in files.items() if v["num_entries"] is not None}
        maybe_base_form = dsprep.form

        if maybe_base_form is not None:
            maybe_base_form = ak.forms.from_json(decompress_form(maybe_base_form))
        events, report = getEvents(files, known_form=maybe_base_form, cache=self.cache)



        daskres = DatasetDaskRunResult(dsprep, {}, {}, {}, report, None)
        dataset_analyzer = DatasetProcessor(
            daskres,
            dsprep.dataset_input.dataset_name,
            dsprep.dataset_input.fill_name,
            dsprep.dataset_input.last_ancestor,
            dsprep.dataset_input.profile,
            delayed=delayed,
            skim_save_path=skim_save_path,
        )
        prog_bar_updater(visible=True)

        logger.info("Adding processing info for all modules")
        for m in this_dataset_modules:
            i = m.processing_info or {}
            logger.info(f"Adding processing info for module {m.name}:\n{i}")
            dataset_analyzer.processing_info.update(i)

        logger.info(
            f"Final module processing info is:\n{dataset_analyzer.processing_info}"
        )
        #events=events.compute()
        #dataset_analyzer.delayed=False

        for m in this_dataset_modules:
            logger.info(f"Adding module {m.name} to dataset {dataset_name}")
            test = m(events, dataset_analyzer)
            events, dataset_analyzer = test

            prog_bar_updater(advance=1)
        prog_bar_updater(visible=False)

        return daskres, dataset_analyzer.side_effect_computes
