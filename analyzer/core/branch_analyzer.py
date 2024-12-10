import concurrent.futures
import copy
import inspect
import itertools as it
import logging
import pickle as pkl
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import awkward as ak
import dask
import distributed
import yaml
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId, SampleType
from analyzer.utils.file_tools import extractCmsLocation
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form

from .analysis_modules import MODULE_REPO
from .configuration import (
    AnalysisDescription,
    AnalysisStage,
    ExecutionConfig,
    FileConfig,
    loadDescription,
)
from .histograms import HistogramSpec, generateHistogramCollection
from .preprocessed import SamplePreprocessed, preprocessBulk, preprocessRaw
from .results import AnalysisResult, SectorResult, SelectionResult, SubSectorResult
from .sector import SubSector, getParamsForSubSector
from .selection import SelectionManager
from .specifiers import SubSectorId, BranchId
from .weights import WeightManager

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track







class SubSectorAnalyzer:
    def run(self, columns):
        return SubSectorResult

class SampleAnalyzer:
    pass


def analyzeBranch(columns,  SampleAnalyzer:)

@dataclass
class BranchAnalyzer:
    _preselected_events: Any
    _selected_events: Any

    weight_manager: WeightManager = field(default_factory=WeightManager)
    selection_manager: SelectionManager = field(default_factory=SelectionManager)
    categories: defaultdict[str, dict[str, Category]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    branches: list[BranchId] = field(default_factory=list)
    results: dict[SubSectorId, SubSectorResult] = field(default_factory=dict)

    user_execution_options: Optional[ExecutionConfig] = None
    user_file_options: Optional[ExecutionConfig] = None

    def __post_init__(self):
        self.subsectors = self._getSubSectors()
        self.client = distributed.client._get_global_client()

    def getExecOpts(self):
        return ExecutionConfig(
            **self.description.execution_config.model_dump(),
            **(
                self.user_execution_options.model_dump()
                if self.user_execution_options is not None
                else {}
            ),
        )

    def getFileOpts(self):
        return FileConfig(
            **self.description.file_config.model_dump(),
            **(
                self.user_file_options.model_dump()
                if self.user_file_options is not None
                else {}
            ),
        )

    def prefillResults(self):
        for subsector in self.subsectors:
            subsector_id = subsector.subsector_id
            params = getParamsForSubSector(
                subsector_id, self.dataset_repo, self.era_repo
            )
            self.results[subsector_id] = SubSectorResult(params=params)

    def dropEmptySubSectors(self):
        new = [
            x
            for x in self.subsectors
            if x.subsector_id.sample_id in self.preprocessed_samples
            and self.preprocessed_samples[x.subsector_id.sample_id].chunks
        ]
        logger.warn(f"Dropping {len(self.subsectors) - len(new)} empty subsectors.")
        self.subsectors = new

    def preprocessDatasets(self, **kwargs):
        step_size = self.getExecOpts().step_size
        loc_prio = self.getFileOpts().location_priority_regex
        logger.info(f"Preprocessing samples with chunk size {step_size}")
        logger.info(f"Preprocessing with location priority:\n{loc_prio}")

        r = preprocessBulk(
            self.dataset_repo,
            set(x.subsector_id.sample_id for x in self.subsectors),
            step_size=step_size,
            file_retrieval_kwargs={"location_priority_regex": loc_prio},
        )
        self.preprocessed_samples = {x.sample_id: x for x in r}

    @staticmethod
    def __loadOne(sample_id, files, maybe_base_form):
        logger.debug(f"Loading events for sample {sample_id}.")
        num_tries = 0
        max_tries = 4
        while num_tries < max_tries:
            num_tries += 1
            try:
                logger.info(f"Loading events from {sample_id}")
                events, report = NanoEventsFactory.from_root(
                    files,
                    schemaclass=NanoAODSchema,
                    uproot_options=dict(
                        allow_read_errors_with_report=True,
                        timeout=30,
                    ),
                    known_base_form=maybe_base_form,
                ).events()
                break
            except Exception as e:
                logger.warn(f"Error while loading file:\n{e}")
                traceback = e.__traceback__
                frames = inspect.getinnerframes(traceback)
                print(frames)
                print(frames[3])
                if num_tries == max_tries:
                    logger.error(f"Critical error while loading file:\n{e}")
                    events, report = None, None
                    # raise
        return sample_id, events, report

    def loadEvents(self, **kwargs):
        loc_prio = self.getFileOpts().location_priority_regex
        logger.info(f"Preprocessing with location priority:\n{loc_prio}")
        futures = []
        # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        samples = set(x.subsector_id.sample_id for x in self.subsectors)
        for sample_id in samples:
            spre = self.preprocessed_samples[sample_id]
            ds = spre.getCoffeaDataset(
                self.dataset_repo, location_priority_regex=loc_prio
            )
            if spre.form is not None:
                maybe_base_form = ak.forms.from_json(decompress_form(spre.form))
            else:
                maybe_base_form = None
            # f = executor.submit(
            #     Analyzer.__loadOne, sample_id, ds["files"], maybe_base_form
            # )
            f = Analyzer.__loadOne(sample_id, ds["files"], maybe_base_form)
            futures.append(f)

            # to_iter = concurrent.futures.as_completed(futures)
        to_iter = futures
        if CONFIG.PRETTY_MODE:
            to_iter = track(to_iter, description="Loading Events", total=len(futures))
        for future in to_iter:
            logger.debug(f"Done loading events for sample {sample_id}.")
            # r = future.result()
            r = future
            if r[1] is not None:
                self._sample_events[r[0]] = r[1]
                self._sample_reports[r[0]] = r[2]

    def _getSubSectors(self):
        s_pairs = []
        ret = []
        for dataset_name, regions in self.description.samples.items():
            if isinstance(regions, str) and regions == "All":
                regions = [r.name for r in self.description.regions]
            for r in regions:
                s_pairs.append((dataset_name, r))
        for dataset_name, region_name in s_pairs:
            dataset = self.dataset_repo[dataset_name]
            region = self.description.getRegion(region_name)
            for sample in dataset.samples:
                subsector = SubSector.fromRegion(
                    region, sample, MODULE_REPO, self.era_repo
                )
                logger.debug(f"Registered subsector {subsector.subsector_id}")
                ret.append(subsector)

        return ret

    def addWeight(self, subsector_id, name, central, variations=None):
        logger.debug(
            f'SubSector[{subsector_id}] adding weight "{name}" with {len(variations or [])} variations.'
        )
        varia = variations or {}
        self.weight_manager.add(subsector_id, name, central, varia)

    def addCategory(self, subsector_id, category):
        logger.debug(f'SubSector[{subsector_id}] adding category "{category.name}"')
        self.categories[subsector_id][category.name] = category

    def addSelection(self, subsector_id, name, mask, type="and", stage="preselection"):
        self.selection_manager.register(
            subsector_id, name, mask, type=type, stage=stage
        )

    def _populateCutflow(self, subsector_id):
        cf = self.selection_manager.getCutflow(subsector_id)
        is_mc = (
            getParamsForSubSector(
                subsector_id, self.dataset_repo, self.era_repo
            ).sector.dataset.sample_type
            == SampleType.MC
        )
        if is_mc:
            weighted_sum = self.weight_manager.totalWeight(subsector_id)
        else:
            weighted_sum = None

        raw = ak.num(self._subsector_events[subsector_id], axis=0)

        self.results[subsector_id].cutflow_data = SelectionResult(
            cutflow=cf, raw_passed=raw, weighted_sum=weighted_sum
        )

    def getTotalMcWeights(self):
        def totalGen(ds_name, sample_name, e):
            ds = self.dataset_repo[ds_name]
            if not ds.sample_type == SampleType.MC:
                return None
            if ds.skimmed_from:
                return totalGen(ds.skimmed_from, sample_name, e)
            return ds.getSample(sample_name).n_events
            # if ds.sample_type == SampleType.MC:
            #    return ak.sum(e.genWeight, axis=0)
            # else:
            #    return None

        ret = {
            sample_id: totalGen(sample_id.dataset_name, sample_id.sample_name, e)
            for sample_id, e in self._sample_events.items()
        }
        return {x: y for x, y in ret.items() if y is not None}

    def makeHistogramFromSpec(
        self, subsector_id, spec: HistogramSpec, values, mask=None
    ):
        if (
            getParamsForSubSector(
                subsector_id, self.dataset_repo, self.era_repo
            ).sector.dataset.sample_type
            == SampleType.Data
        ):
            subsector_weighter = None
        else:
            subsector_weighter = self.weight_manager.getSubSectorWeighter(subsector_id)
        hc = generateHistogramCollection(
            spec,
            values,
            list(self.categories[subsector_id].values()),
            subsector_weighter,
            mask=mask,
        )
        self.results[subsector_id].histograms[spec.name] = hc

    def makeHistogram(
        self,
        subsector_id,
        name,
        axes,
        values,
        variations=None,
        weights=None,
        description="",
        no_scale=False,
        mask=None,
        storage="weight",
    ):
        if (
            getParamsForSubSector(
                subsector_id, self.dataset_repo, self.era_repo
            ).sector.dataset.sample_type
            == SampleType.Data
        ):
            variations = []
            weights = []
        else:
            if weights is None:
                weights = self.weight_manager.weight_names(subsector_id)

            if variations is None:
                variations = self.weight_manager.variations(subsector_id)

        logger.debug(f"Creating histogram {weights}")

        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        spec = HistogramSpec(
            name=name,
            axes=axes,
            storage=storage,
            description=description,
            weights=weights,
            variations=variations,
            no_scale=no_scale,
        )
        self.makeHistogramFromSpec(subsector_id, spec, values, mask=mask)

    def __getStageProcessor(self, subsector_id, stage):
        mapping = {
            AnalysisStage.Preselection: SubSectorAnalyzer.Selector(
                self, subsector_id, AnalysisStage.Preselection
            ),
            AnalysisStage.Selection: SubSectorAnalyzer.Selector(
                self, subsector_id, AnalysisStage.Selection
            ),
            AnalysisStage.Categorization: SubSectorAnalyzer.Categorizer(
                self, subsector_id
            ),
            AnalysisStage.Weights: SubSectorAnalyzer.Weighter(self, subsector_id),
            AnalysisStage.Histogramming: SubSectorAnalyzer.Histogrammer(
                self, subsector_id
            ),
        }
        return mapping.get(stage)

    def _applySubSectorToEvents(self, subsector, stage):
        subsector_id = subsector.subsector_id
        sp = self.__getStageProcessor(subsector_id, stage)
        mapping = {
            AnalysisStage.Preselection: subsector.preselection,
            AnalysisStage.Selection: subsector.selection,
            AnalysisStage.Correction: subsector.correction,
            AnalysisStage.Categorization: subsector.categories,
            AnalysisStage.Histogramming: subsector.histograms,
            AnalysisStage.ObjectDefinition: subsector.objects,
            AnalysisStage.Weights: subsector.weights,
        }
        if stage in [AnalysisStage.Preselection]:
            events = self._sample_events[subsector_id.sample_id]
        elif stage in [AnalysisStage.Selection, AnalysisStage.ObjectDefinition, AnalysisStage.Correction]:
            events = self._preselected_events[subsector_id.sample_id]
        else:
            events = self._subsector_events[subsector_id]
        params = getParamsForSubSector(subsector_id, self.dataset_repo, self.era_repo)
        for module in mapping[stage]:
            if sp:
                module(events, params, sp)
            else:
                module(events, params)

    def runStage(self, stage):
        for subsector in self.subsectors:
            self._applySubSectorToEvents(subsector, stage)

    def applyPreselections(self):
        for sample_id, events in self._sample_events.items():
            self._preselected_events[
                sample_id
            ] = self.selection_manager.maskPreselection(sample_id, events)

    def applySelection(self):
        self.selection_manager.addPreselectionMasks()
        for subsector in self.subsectors:
            subsector_id = subsector.subsector_id
            events = self._preselected_events[subsector_id.sample_id]
            e = self.selection_manager.maskSubSector(subsector_id, events)
            self._subsector_events[subsector_id] = e

    def populateCutflows(self):
        for subsector in self.subsectors:
            self._populateCutflow(subsector.subsector_id)


def buildAnalysisResult(analyzer):

    if not analyzer.preprocessed_samples:
        logger.info(f"Analyzer does not already have preprocessed samples.")
        analyzer.preprocessDatasets()
    # When patching results, we may have subsectors with no preprocessing object
    # since all the events have already been processed succesfully.
    analyzer.dropEmptySubSectors()
    analyzer.prefillResults()
    logger.info(f"Running analysis with {len(analyzer.subsectors)} subsectors.")
    analyzer.loadEvents()

    mc_weights = analyzer.getTotalMcWeights()
    analyzer.runStage(AnalysisStage.Preselection)

    analyzer.applyPreselections()

    analyzer.runStage(AnalysisStage.Correction)

    analyzer.runStage(AnalysisStage.ObjectDefinition)
    analyzer.runStage(AnalysisStage.Selection)

    analyzer.applySelection()

    analyzer.runStage(AnalysisStage.Weights)

    analyzer.populateCutflows()

    analyzer.runStage(AnalysisStage.Categorization)
    analyzer.runStage(AnalysisStage.Histogramming)

    logger.info(f"Finished lazy results construction")
    # processed_chunks={
    #     s: getProcessedChunks(1, analyzer._sample_reports[s])
    #     for s in analyzer._sample_reports
    # },

    return AnalysisResult(
        description=analyzer.description,
        preprocessed_samples=analyzer.preprocessed_samples,
        processed_chunks=analyzer._sample_reports,
        results=analyzer.results,
        total_mc_weights=mc_weights,
    )
