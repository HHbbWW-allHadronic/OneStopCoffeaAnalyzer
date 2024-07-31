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
from .lumi import getLumiMask
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
            ]
            for x, _ in futures
        },
        "side_effects": [y for _, y in futures],
    }
    computed, *rest = dask.compute(dsk, retries=3)

    ret = {
        name: DatasetRunResult(prep, h, getProcessedChunks(rep), nsh, nshl)
        for name, (prep, h, nsh, nshl, rep) in computed["main"].items()
    }

    return ret


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
#         name: DatasetRunResult(prep, h, r, nsh, nshl, rep)
#         for name, (prep, h, nsh, nshl, r, rep) in computed.items()
#     }

@dask.delayed
def createFutureResult(modules, prepped_dataset):
    dataset_name = prepped_dataset.dataset_input.dataset_name
    logger.debug(f"Generating futures for dataset {dataset_name}")
    files = prepped_dataset.getCoffeaDataset()["files"]
    maybe_base_form = prepped_dataset.coffea_dataset_split.get("form", None)
    if maybe_base_form is not None:
        maybe_base_form = ak.forms.from_json(decompress_form(maybe_base_form))
    events, report = NanoEventsFactory.from_root(
        files,
        schemaclass=NanoAODSchema,
        uproot_options=dict(
            allow_read_errors_with_report=True,
            use_threads=False,
        ),
        known_base_form=maybe_base_form,
    ).events()
    daskres = DatasetDaskRunResult(prepped_dataset, {}, {}, {}, report)
    dataset_analyzer = DatasetProcessor(
        daskres, prepped_dataset.dataset_input.fill_name
    )
    for m in modules:
        logger.info(f"Adding module {m.name} to dataset {dataset_name}")
        test = m(events, dataset_analyzer)
        events, dataset_analyzer = test

    return daskres


class Analyzer:
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
        self,
        dsprep: DatasetPreprocessed,
        delayed: bool = True,
        skim_save_path: str = None,
        prog_bar_updater=None,
        file_retrieval_kwargs=None,
    ) -> DatasetDaskRunResult:
        if file_retrieval_kwargs is None:
            file_retrieval_kwargs = {}

        dataset_name = dsprep.dataset_input.dataset_name
        lumi_json = dsprep.dataset_input.lumi_json
        logger.debug(f"Generating futures for dataset {dataset_name}")
        files = dsprep.getCoffeaDataset(**file_retrieval_kwargs)["files"]
        files = {k: v for k, v in files.items() if v["num_entries"] is not None}
        maybe_base_form = dsprep.form

        if maybe_base_form is not None:
            maybe_base_form = ak.forms.from_json(decompress_form(maybe_base_form))
        events, report = getEvents(files, known_form=maybe_base_form, cache=self.cache)

        if lumi_json:
            logger.info(f'Dataset {dataset_name}: Using lumi json file "{lumi_json}".')
            lmask = getLumiMask(lumi_json)
            events = events[lmask(events.run, events.luminosityBlock)]

        daskres = DatasetDaskRunResult(dsprep, {}, {}, {}, report)
        dataset_analyzer = DatasetProcessor(
            daskres,
            dsprep.dataset_input.dataset_name,
            dsprep.dataset_input.fill_name,
            dsprep.dataset_input.profile,
            delayed=delayed,
            skim_save_path=skim_save_path,
        )
        prog_bar_updater(visible=True)

        for m in self.modules:
            logger.info(f"Adding module {m.name} to dataset {dataset_name}")
            test = m(events, dataset_analyzer)
            events, dataset_analyzer = test

            prog_bar_updater(advance=1)
        prog_bar_updater(visible=False)

        return daskres, dataset_analyzer.side_effect_computes
