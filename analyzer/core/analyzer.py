import itertools as it
import logging
from dataclasses import dataclass, field
from typing import Any, Union


from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId
from coffea.nanoevents import NanoAODSchema
from .region_analyzer import RegionAnalyzer, getParamsSample
from .configuration import loadDescription, getSubSectors
from .columns import Columns
from .selection import SelectionSet
import analyzer.core.results as results
import analyzer.core.executor as executor

if CONFIG.PRETTY_MODE:
    from rich import print

logger = logging.getLogger(__name__)


@dataclass
class Category:
    name: str
    axis: Any
    values: Any
    distinct_values: set[Union[int, str, float]] = field(default_factory=set)


@dataclass
class Analyzer:
    region_analyzers: list[RegionAnalyzer]

    def _runBranch(
        self,
        region_analyzers,
        columns,
        params,
        preselection,
        preselection_set,
        histogram_storage,
        cutflow_storage,
        variation=None,
    ):
        logger.info(f"Running analysis branch for variation {variation}")
        selection_set = SelectionSet(parent_selection=preselection)
        columns = columns.withSyst(variation)
        for ra in region_analyzers:
            ra.runObjects(columns, params)
        ret = {}
        for ra in region_analyzers:
            hs = histogram_storage[ra.region_name]
            selection = ra.runSelection(columns, params, selection_set)
            if variation is None:
                cutflow_storage[ra.region_name] = selection.getSelectionFlow()

            mask = selection.getMask()
            new_cols = columns.withEvents(columns.events[mask])
            res = ra.runPostSelection(new_cols, params, hs)
            ret[(ra.region_name, variation)] = res
        return ret

    def _runPreselectionGroup(
        self, events, params, region_analyzers, preselection, preselection_set
    ):

        histogram_storage = {ra.region_name: {} for ra in region_analyzers}
        cutflow_storage = {ra.region_name: {} for ra in region_analyzers}
        mask = preselection.getMask()
        events = events[mask]
        columns = Columns(events)
        for ra in region_analyzers:
            ra.runCorrections(events, params, columns)

        branches = [None] + columns.allShapes()
        logger.info(f"Known variations are {branches}")
        ret = {}
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
            ra.region_name: results.CoreSubSectorResult(
                region=ra.model_dump(),
                params=params,
                histograms=histogram_storage[ra.region_name],
                other_data={},
                selection_flow=cutflow_storage[ra.region_name],
            )
            for ra in region_analyzers
        }
        return ret

    def run(self, events, params):
        preselection_set = SelectionSet()
        region_preselections = []
        for analyzer in self.region_analyzers:
            region_preselections.append(
                (analyzer, analyzer.runPreselection(events, params, preselection_set))
            )
        k = lambda x: x[1].names

        presel_regions = it.groupby(sorted(region_preselections, key=k), key=k)
        presel_regions = {x: list(y) for x, y in presel_regions}
        ret = {}
        for presels, items in presel_regions.items():
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


def makeUnits(subsectors, dataset_repo, era_repo, file_retrieval_kwargs):
    ret = []
    for sample_id, region_analyzers in subsectors.items():
        params = dataset_repo[sample_id].params
        params.dataset.populateEra(era_repo)
        u = executor.ExecutionUnit(
            sample_id=sample_id,
            sample_params=params,
            file_set=dataset_repo[sample_id].getFileSet(file_retrieval_kwargs),
            analyzer=Analyzer(region_analyzers),
        )
        ret.append(u)
    return ret


if __name__ == "__main__":
    import analyzer.modules
    from analyzer.logging import setup_logging

    setup_logging()

    d = loadDescription("configurations/data_mc_comp.yaml")

    dr = DatasetRepo.getConfig()
    er = EraRepo.getConfig()

    NanoAODSchema.warn_missing_crossrefs = False

    subsectors = getSubSectors(d, dr, er)
    units = makeUnits(subsectors, dr, er, {})

    de = executor.DaskExecutor()
    de.run(units)
    # ret = de.run(
    #     analyzer, sample_params, {"test1.root": "Events", "test2.root": "Events"}
    # )
