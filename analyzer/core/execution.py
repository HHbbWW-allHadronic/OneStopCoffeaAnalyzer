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

import awkward as ak
import dask
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from distributed import Client, get_client, rejoin, secede

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
            x.raw_events_processed,
            x.run_report,
        ]
        for x in futures
    }

    optims = [
        (
            lambda dsk, keys: dask.optimization.inline(
                dsk, [x for x in keys if "phi" in x]
            )
        )
    ]
    optims=[]

    if client is None:
        computed, *rest = dask.compute(dsk, scheduler="single-threaded")
    else:
        f = client.compute(
            dsk,
            optimizations=optims,
        )
        computed = client.gather(f)

    return {
        name: DatasetRunResult(prep, h, r, rep)
        for name, (prep, h, r, rep) in computed.items()
    }


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
        ),
        known_base_form=maybe_base_form,
    ).events()
    daskres = DatasetDaskRunResult(prepped_dataset, {}, ak.num(events, axis=0), report)
    dataset_analyzer = DatasetProcessor(
        daskres, prepped_dataset.dataset_input.fill_name
    )
    for m in modules:
        logger.info(f"Adding module {m.name} to dataset {dataset_name}")
        test = m(events, dataset_analyzer)
        events, dataset_analyzer = test

    return daskres


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
        self.__dataset_ps: Dict[str, DatasetProcessingState] = {}
        self.__run_reports: Dict[str, dak.Array] = {}

    def __createAndSortModules(self, *module_names):
        m = namesToModules(module_names)
        t = generateTopology(m)
        modules = sortModules(m)
        return modules

    def getDatasetFutures(
        self, dsprep: DatasetPreprocessed, delayed=True
    ) -> DatasetDaskRunResult:
        # return createFutureResult(self.modules, dsprep)
        dataset_name = dsprep.dataset_input.dataset_name
        lumi_json = dsprep.dataset_input.lumi_json
        logger.debug(f"Generating futures for dataset {dataset_name}")
        files = dsprep.getCoffeaDataset()["files"]
        maybe_base_form = dsprep.coffea_dataset_split.get("form", None)
        if maybe_base_form is not None:
            maybe_base_form = ak.forms.from_json(decompress_form(maybe_base_form))
        events, report = NanoEventsFactory.from_root(
            files,
            schemaclass=NanoAODSchema,
            uproot_options=dict(
                allow_read_errors_with_report=True,
            ),
            known_base_form=maybe_base_form,
            persistent_cache=self.cache,
        ).events()

        if lumi_json:
            logger.info(f'Dataset {dataset_name}: Using lumi json file "{lumi_json}".')
            lmask = getLumiMask(lumi_json)
            events = events[lmask(events.run, events.luminosityBlock)]

        if delayed:
            daskres = DatasetDaskRunResult(dsprep, {}, ak.num(events, axis=0), report)
        else:
            events = events.compute()
            report = report.compute()
            daskres = DatasetRunResult(dsprep, {}, ak.num(events, axis=0), report)

        dataset_analyzer = DatasetProcessor(
            daskres, dsprep.dataset_input.fill_name, delayed=delayed
        )
        for m in self.modules:
            logger.info(f"Adding module {m.name} to dataset {dataset_name}")
            test = m(events, dataset_analyzer)
            events, dataset_analyzer = test
        return daskres
