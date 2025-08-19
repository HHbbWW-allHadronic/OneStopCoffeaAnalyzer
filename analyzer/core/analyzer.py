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
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)


def callTimeout(func, timeout_seconds, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            raise TimeoutError("Function execution timed out!")


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
        ret = {}
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
            res = ra.runPostSelection(new_cols, params, hs)
            ret[(ra.region_name, variation)] = res
        return ret

    def _runPreselectionGroup(
        self, events, params, region_analyzers, preselection, preselection_set
    ):
        # Each region has its own histograms
        histogram_storage = {ra.region_name: {} for ra in region_analyzers}
        # Each region has its own cutflow
        cutflow_storage = {ra.region_name: {} for ra in region_analyzers}
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
        ret = {}
        # Run over each shape variation in the analysis
        for variation in branches:
            logger.info(f'Running branch for variation "{variation}"')
            res = self._runBranch(
                region_analyzers,
                columns,
                params,
                preselection,
                preselection_set,
                histogram_storage,
                cutflow_storage,
                variation=variation,
            )
            ret.update(res)

        ret = {
            ra.region_name: results.SubSectorResult(
                region=ra,
                base_result=results.BaseResult(
                    histograms=histogram_storage[ra.region_name],
                    other_data={},
                    selection_flow=cutflow_storage[ra.region_name],
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
        return ret

    def runDelayed(self, *args, **kwargs):
        return dask.delayed(self.run)(*args, **kwargs)

    def runChunks(
        self,
        fileset,
        params,
        known_form=None,
        treepath="Events",
        processing_timeout=60,
        load_timeout=10,
    ):
        try:
            from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

            chunks = fileset.toCoffeaDataset()["files"]

            events = callTimeout(
                NanoEventsFactory.from_root,
                load_timeout,
                chunks,
                schemaclass=NanoAODSchema,
                known_base_form=known_form,
                delayed=True,
                uproot_options=dict(use_threads=False),
            ).events()

            result = results.subsector_adapter.dump_python(self.run(events, params))
            result = callTimeout(
                dask.compute, processing_timeout, result, scheduler="synchronous"
            )[0]
            result = results.subsector_adapter.validate_python(result)
            return (fileset, result)  # self.run(events, params))

        except Exception:
            return None

    def ensureFunction(self, module_repo):
        for ra in self.region_analyzers:
            ra.ensureFunction(module_repo)
