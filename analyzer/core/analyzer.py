import itertools as it
import dask
import logging
from dataclasses import dataclass
from collections.abc import Sequence


from analyzer.core.region_analyzer import RegionAnalyzer

from analyzer.datasets import SampleParams
from analyzer.core.columns import Columns
from analyzer.core.selection import SelectionSet, Selection, SelectionFlow
import analyzer.core.results as results
from analyzer.core.histograms import Hist
from concurrent.futures import ProcessPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)


def callTimeout(process_timeout, function, *args, **kwargs):
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(function, *args, **kwargs)
            return future.result(timeout=process_timeout)
        except TimeoutError:
            raise
            # for pid, process in executor._processes.items():
            #     process.terminate()


def checkProcess(process, max_mem):
    try:
        memory = psutil.Process(process.pid).memory_info().rss
    except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
        return

    if memory / self.memory_limit <= self.memory_terminate_fraction:
        return
    else:
        process.terminate()


@dataclass
class Analyzer:
    """
    Contains a list of RegionAnalyzers, which are run in a way to limit duplicated processing.
    """

    region_analyzers: list[RegionAnalyzer]

    def _runBranch(
        self,
        region_analyzers: Sequence[RegionAnalyzer],
        columns: Columns,
        params: SampleParams,
        preselection: Selection,
        preselection_set: SelectionSet,
        histogram_storage: dict[str, Hist],
        cutflow_storage: dict[str, SelectionFlow],
        weight_storage: dict[str, float],
        variation=None,
    ) -> dict[tuple[str, str], results.SubSectorResult]:
        """
        Run a "Branch" for a given shape variation
        """
        logger.info(f"Running analysis branch for variation {variation}")
        selection_set = SelectionSet(parent_selection=preselection)
        # Set columns to use variation
        columns = columns.withSyst(variation)
        for ra in region_analyzers:
            ra.runObjects(columns, params)
        for ra in region_analyzers:
            hs = histogram_storage[ra.region_name]
            selection = ra.runSelection(columns, params, selection_set)
            if variation is None:
                # Only save cutflow if we are on the nominal path
                cutflow_storage[ra.region_name] = selection.getSelectionFlow()

            mask = selection.getMask()
            # Only apply mask if there is something to mask
            if mask is not None:
                new_cols = columns.withEvents(columns.events[mask])
            else:
                new_cols = columns
            # Once the final selection has been performed for a region
            # we can run the remainder of the analyzer
            if variation is None:
                ra.runPostSelection(
                    new_cols, params, hs, weight_storage[ra.region_name]
                )
            else:
                ra.runPostSelection(new_cols, params, hs)

    def _runPreselectionGroup(
        self, events, params, region_analyzers, preselection, preselection_set
    ):
        # Each region has its own histograms
        histogram_storage = {ra.region_name: {} for ra in region_analyzers}
        # Each region has its own cutflow
        cutflow_storage = {ra.region_name: {} for ra in region_analyzers}
        # Each region has its own weights
        weight_storage = {ra.region_name: {} for ra in region_analyzers}

        mask = preselection.getMask()
        if mask is not None:
            events = events[mask]
        # Events are wrapped to allow for dyanamic selection of columns
        # based on the active shape systematic
        columns = Columns(events)
        for ra in region_analyzers:
            # Generate corrections, at this point the analyzer will have
            # branches corresponding to shape variations
            ra.runCorrections(events, params, columns)

        branches = [None] + columns.allShapes()
        logger.info(f"Known variations are {branches}")
        # Run over each shape variation in the analysis
        for variation in branches:
            logger.info(f'Running branch for variation "{variation}"')
            self._runBranch(
                region_analyzers,
                columns,
                params,
                preselection,
                preselection_set,
                histogram_storage,
                cutflow_storage,
                weight_storage,
                variation=variation,
            )

        ret = {
            ra.region_name: results.SubSectorResult(
                region=ra,
                base_result=results.BaseResult(
                    histograms=histogram_storage[ra.region_name],
                    other_data={},
                    selection_flow=cutflow_storage[ra.region_name],
                    post_sel_weight_flow=weight_storage[ra.region_name],
                ),
            )
            for ra in region_analyzers
        }
        return ret

    def run(self, events, params):
        """
        Run analyzer over a collection of events, given some parameters.
        """
        preselection_set = SelectionSet()
        region_preselections = []
        for analyzer in self.region_analyzers:
            # Get the analyzer and the preselection for each region
            region_preselections.append(
                (analyzer, analyzer.runPreselection(events, params, preselection_set))
            )
        k = lambda x: x[1].names

        # Group regions that share the same preselection.
        presel_regions = it.groupby(sorted(region_preselections, key=k), key=k)
        presel_regions = {x: list(y) for x, y in presel_regions}
        ret = {}
        for presels, items in presel_regions.items():
            # Separately run each collection of regions sharing a preselection
            logger.info(
                f'Running over preselection region "{presels}" containing "{len(items)}" regions.'
            )
            sel = items[0][1]
            ret.update(
                self._runPreselectionGroup(
                    events, params, [x[0] for x in items], sel, preselection_set
                )
            )
        return results.MultiSectorResult(ret)

    def runDelayed(self, *args, **kwargs):
        return dask.delayed(self.run)(*args, **kwargs)

    def ensureFunction(self, module_repo):
        for ra in self.region_analyzers:
            ra.ensureFunction(module_repo)


def runAnalyzerChunks(
    analyzer,
    fileset,
    params,
    known_form=None,
    treepath="Events",
    timeout=120,
):
    try:
        if timeout:
            logger.info(f"Starting run of analyzer using file set: {fileset}")
            return callTimeout(
                timeout,
                runAnalyzerChunks,
                analyzer,
                fileset,
                params,
                known_form=known_form,
                treepath=treepath,
                timeout=None,
            )
        else:
            from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
            from analyzer.logging import setup_logging

            setup_logging()

            chunks = fileset.toCoffeaDataset()["files"]

            logger.info(f"Loading events from {fileset}")
            events = NanoEventsFactory.from_root(
                chunks,
                schemaclass=NanoAODSchema,
                known_base_form=known_form,
                delayed=True,
                uproot_options=dict(use_threads=False),
            ).events()

            result = analyzer.run(events, params)
            result = result.model_dump()
            result = dask.compute(result, scheduler="threads")[0]
            logger.info(f"Analysis completed successfully for {fileset}")
            result = results.MultiSectorResult(**result)

            return (fileset, result)

    except Exception as e:
        return e
