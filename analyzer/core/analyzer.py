# from __future__ import annotations
import concurrent.futures
import copy
import functools as ft
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import awkward as ak
import dask
import pydantic as pyd
import yaml
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId, SampleType
from analyzer.utils.file_tools import extractCmsLocation
from analyzer.utils.structure_tools import accumulate
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from distributed import Client
from rich import print

from .analysis_modules import MODULE_REPO
from .common_types import Scalar
from .configuration import AnalysisDescription, AnalysisStage
from .histograms import HistogramCollection, HistogramSpec, generateHistogramCollection
from .preprocessed import SamplePreprocessed, preprocessBulk
from .sector import Sector, SectorId, SectorParams, getParamsForSector
from .selection import Cutflow, SelectionManager
from .weights import WeightManager

if CONFIG.PRETTY_MODE:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
    )

logger = logging.getLogger("analyzer.core")


def getSectionHash(am_descs):
    to_hash = [{"name": x.name, "configuration": x.configuration} for x in am_descs]
    return hash(json.dumps(to_hash, sort_keys=True))


@dataclass
class Category:
    name: str
    axis: Any
    values: Any
    distinct_values: set[Union[int, str, float]] = field(default_factory=set)


class SampleCutflow(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    cutflow: Cutflow
    raw_passed: Scalar
    weighted_sum: Optional[tuple[Scalar, Scalar]]

    def __add__(self, other):
        """Two cutflows may be sumed by simply adding them"""

        return SampleCutflow(
            cutflow=self.cutflow + other.cutflow,
            raw_passed=self.raw_passed + other.raw_passed,
            weighted_sum=(
                (self.weighted_sum[0] + other.weighted_sum[0]),
                (self.weighted_sum[1] + other.weighted_sum[1]),
            ),
        )

    def scaled(self, scale):
        if self.weighted_sum:
            nws = (
                self.weighted_sum[0] * scale,
                self.weighted_sum[1] * (scale**2),
            )
        else:
            nws = None
        return SampleCutflow(
            cutflow=self.cutflow,
            raw_passed=self.raw_passed,
            weighted_sum=nws,
        )


class SectorResult(pyd.BaseModel):
    params: SectorParams
    histograms: dict[str, HistogramCollection] = pyd.Field(default_factory=dict)
    other_data: dict[str, Any] = pyd.Field(default_factory=dict)
    cutflow_data: Optional[SampleCutflow] = None

    def __add__(self, other):
        """Two sector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.params != other.params:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return SectorResult(
            params=self.params,
            histograms=new_hists,
            other_data=new_other,
            cutflow_data=self.cutflow_data + other.cutflow_data,
        )

    def scaled(self, scale):
        return SectorResult(
            params=self.params,
            histograms={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            cutflow_data=self.cutflow_data.scaled(scale),
        )


class DatasetResult(pyd.BaseModel):
    dataset_params: dict[str, Any]
    era_params: dict[str, Any]
    histograms: dict[str, HistogramCollection]
    other_data: dict[str, Any]
    cutflow_data: Optional[SampleCutflow]

    @staticmethod
    def fromSectorResult(sector_result):
        return DatasetResult(
            dataset_params=sector_result.params.dataset_params,
            era_params=sector_result.params.era_params,
            histograms=sector_result.histograms,
            other_data=sector_result.other_data,
            cutflow_data=sector_result.cutflow_data,
        )

    def scaled(self, scale):
        return DatasetResult(
            dataset_params=self.dataset_params,
            era_params=self.era_params,
            histogrmams={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            cutflow_data=self.cutflow_data.scaled(scale),
        )

    def __add__(self, other):
        """Two sector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if (
            self.dataset_params != other.dataset_params
            or self.era_params != other.era_params
        ):
            raise RuntimeError(f"Error: Attempting to merge incomaptible results")
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return DatasetResult(
            dataset_params=self.dataset_params,
            era_params=self.era_params,
            histograms=new_hists,
            other_data=new_other,
            cutflow_data=self.cutflow_data + other.cutflow_data,
        )


class AnalysisResult(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)
    description: AnalysisDescription
    preprocessed_samples: dict[SampleId, SamplePreprocessed]
    processed_chunks: dict[SampleId, Any]
    results: dict[SectorId, SectorResult]
    total_mc_weights: dict[SampleId, Optional[Scalar]]

    @property
    def raw_events_processed(self):
        return {
            sample_id: sum(e - s for _, s, e in chunks)
            for sample_id, chunks in self.results
        }

    def getBadChunks(self):
        ret = {
            self.preprocessed_samples[n].difference(self.processed_chunks[n])
            for n in self.preprocessed_samples
        }
        return ret

    def getMissingPreprocessed(self):
        logger.debug(f"")
        prepped = copy.deepcopy(self.preprocessed_samples)
        bad_chunks = self.getBadChunks()
        for p in prepped:
            logger.debug(f"Sample {p.sample_id} has {len(bad_chunks[p])} bad chunks.")
            prepped[p].limit_chunks = bad_chunks[p]
        return prepped

    def __add__(self, other):
        """Add two analysis results together.
        Two results may be added if they come from the same configuration, and do not have any overlapping
        chunks.
        """

        if self.description != other.description:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )

        if any(
            self.processed_chunks.get(x, set()).intersect(
                other.processed_chunk.get(x, set())
            )
            for x in set(self.processed_chunks) | set(other.processed_chunks)
        ):
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )

        desc = copy.copy(self.description)
        new_results = accumulate([self.results, other.results])
        return AnalysisResult(
            description=self.description,
            preprocessed_samples=self.preprocessed_samples,
            processed_chunks=self.processed_chunks,
            results=new_results,
        )

    def getResults(self):
        scaled_sample_results = defaultdict(list)
        for sector_id, result in self.results.items():
            k = (sector_id.sample_id.dataset_name, sector_id.region_name)
            if result.params.sample_type == SampleType.MC:
                result = result.scaled(1 / self.total_mc_weights[sector_id.sample_id])

            scaled_sample_results[k].append(DatasetResult.fromSectorResult(result))

        return {x: ft.reduce(sum, y) for x, y in scaled_sample_results.items()}


class SectorAnalyzer:
    class Selector:
        def __init__(self, parent, sector_id, stage):
            self.parent = parent
            self.stage = stage
            self.sector_id = sector_id

        def add(self, name, mask, type="and"):
            return self.parent.addSelection(
                self.sector_id, name, mask, type=type, stage=self.stage
            )

    class Weighter:
        def __init__(self, parent, sector_id):
            self.parent = parent
            self.sector_id = sector_id

        def add(self, *args, **kwargs):
            return self.parent.addWeight(self.sector_id, *args, **kwargs)

    class Categorizer:
        def __init__(self, parent, sector_id):
            self.parent = parent
            self.sector_id = sector_id

        def add(self, *args, **kwargs):
            return self.parent.addCategory(self.sector_id, *args, **kwargs)

    class Histogrammer:
        def __init__(self, parent, sector_id):
            self.parent = parent
            self.sector_id = sector_id

        def addHistogram(self, *args, **kwargs):
            return self.parent.addHistogram(self.sector_id, *args, **kwargs)

        def H(self, *args, **kwargs):
            return self.parent.makeHistogram(self.sector_id, *args, **kwargs)


@dask.delayed
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


@dataclass
class Analyzer:
    description: AnalysisDescription
    dataset_repo: DatasetRepo
    era_repo: EraRepo
    preprocessed_samples: dict[SampleId, SamplePreprocessed] = field(
        default_factory=dict
    )

    _sample_events: dict[SampleId, Any] = field(default_factory=dict)
    _sample_reports: dict[SampleId, Any] = field(default_factory=dict)
    _preselected_events: dict[SampleId, Any] = field(default_factory=dict)
    _sector_events: dict[SectorId, Any] = field(default_factory=dict)

    weight_manager: WeightManager = field(default_factory=WeightManager)
    selection_manager: SelectionManager = field(default_factory=SelectionManager)
    categories: defaultdict[str, dict[str, Category]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    sectors: list[Sector] = field(default_factory=list)

    results: dict[SectorId, SectorResult] = field(default_factory=dict)

    def __post_init__(self):
        self.sectors = self._getSectors()
        self.__prefillResults()

    def __prefillResults(self):
        for sector in self.sectors:
            sector_id = sector.sector_id
            params = getParamsForSector(sector_id, self.dataset_repo, self.era_repo)
            self.results[sector_id] = SectorResult(params=params)

    def dropEmptySectors(self):
        new = [
            x
            for x in self.sectors
            if x.sector_id.sample_id in self.preprocessed_samples
        ]
        logger.debug(f"Dropping {len(self.sectors) - len(new)} empty sectors.")
        self.sectors = new

    def preprocessDatasets(self, **kwargs):
        chunk_size = self.description.general_config.get("chunk_size", 100000)
        file_args = self.description.general_config["file_retrieval"]
        logger.debug(f"Preprocessing samples with chunk size {chunk_size}")
        logger.debug(f"Preprocessing with file retrieval arguments:\n{file_args}")

        r = preprocessBulk(
            self.dataset_repo,
            set(x.sector_id.sample_id for x in self.sectors),
            step_size=chunk_size,
            file_retrieval_kwargs=file_args,
        )
        self.preprocessed_samples = {x.sample_id: x for x in r}

    @staticmethod
    def __loadOne(sample_id, files, maybe_base_form):
        logger.debug(f"Loading events for sample {sample_id}.")
        events, report = NanoEventsFactory.from_root(
            files,
            schemaclass=NanoAODSchema,
            uproot_options=dict(
                allow_read_errors_with_report=True,
                timeout=30,
            ),
            known_base_form=maybe_base_form,
        ).events()
        return sample_id, events, report

    def loadEvents(self, **kwargs):
        file_args = self.description.general_config["file_retrieval"]
        logger.debug(f"Loading with file retrieval arguments:\n{file_args}")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for sample_id, spre in self.preprocessed_samples.items():
                ds = spre.getCoffeaDataset(self.dataset_repo, **file_args)
                if spre.form is not None:
                    maybe_base_form = ak.forms.from_json(decompress_form(spre.form))
                else:
                    maybe_base_form = None
                f = executor.submit(
                    Analyzer.__loadOne, sample_id, ds["files"], maybe_base_form
                )
                futures.append(f)
            for future in concurrent.futures.as_completed(futures):
                logger.debug(f"Done loading events for sample {sample_id}.")
                r = future.result()
                self._sample_events[r[0]] = r[1]
                self._sample_reports[r[0]] = r[2]

    def _getSectors(self):
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
                sector = Sector.fromRegion(region, sample, MODULE_REPO)
                logger.debug(f"Registered sector {sector.sector_id}")
                ret.append(sector)

        return ret

    def addWeight(self, sector_id, name, central, variations=None):
        logger.debug(
            f'Sector[{sector_id}] adding weight "{name}" with {len(variations or [])} variations.'
        )
        varia = variations or {}
        self.weight_manager.add(sector_id, name, central, varia)

    def addCategory(self, sector_id, category):
        logger.debug(f'Sector[{sector_id}] adding category "{category.name}"')
        self.categories[sector_id][category.name] = category

    def addSelection(self, sector_id, name, mask, type="and", stage="preselection"):
        self.selection_manager.register(sector_id, name, mask, type=type, stage=stage)

    def _populateCutflow(self, sector_id):
        cf = self.selection_manager.getCutflow(sector_id)
        is_mc = (
            getParamsForSector(sector_id, self.dataset_repo, self.era_repo).sample_type
            == SampleType.MC
        )
        if is_mc:
            weighted_sum = self.weight_manager.totalWeight(sector_id)
        else:
            weighted_sum = None

        raw = ak.num(self._sector_events[sector_id], axis=0)

        self.results[sector_id].cutflow_data = SampleCutflow(
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
            # if self.dataset_repo[sample_id.dataset_name].sample_type == SampleType.MC:
            #    return ak.sum(e.genWeight, axis=0)
            # else:
            #    return None

        ret = {
            sample_id: totalGen(sample_id.dataset_name, sample_id.sample_name, e)
            for sample_id, e in self._sample_events.items()
        }
        return {x: y for x, y in ret.items() if y is not None}

    def makeHistogramFromSpec(self, sector_id, spec: HistogramSpec, values, mask=None):
        if (
            getParamsForSector(sector_id, self.dataset_repo, self.era_repo).sample_type
            == SampleType.Data
        ):
            sector_weighter = None
        else:
            sector_weighter = self.weight_manager.getSectorWeighter(sector_id)
        hc = generateHistogramCollection(
            spec,
            values,
            list(self.categories[sector_id].values()),
            sector_weighter,
            mask=mask,
        )
        self.results[sector_id].histograms[spec.name] = hc

    def makeHistogram(
        self,
        sector_id,
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
            getParamsForSector(sector_id, self.dataset_repo, self.era_repo).sample_type
            == SampleType.Data
        ):
            variations = []
            weights = []
        else:
            if weights is None:
                weights = self.weight_manager.weight_names(sector_id)

            if variations is None:
                variations = self.weight_manager.variations(sector_id)

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
        self.makeHistogramFromSpec(sector_id, spec, values, mask=mask)

    def __getStageProcessor(self, sector_id, stage):
        mapping = {
            AnalysisStage.Preselection: SectorAnalyzer.Selector(
                self, sector_id, AnalysisStage.Preselection
            ),
            AnalysisStage.Selection: SectorAnalyzer.Selector(
                self, sector_id, AnalysisStage.Selection
            ),
            AnalysisStage.Categorization: SectorAnalyzer.Categorizer(self, sector_id),
            AnalysisStage.Weights: SectorAnalyzer.Weighter(self, sector_id),
            AnalysisStage.Histogramming: SectorAnalyzer.Histogrammer(self, sector_id),
        }
        return mapping.get(stage)

    def _applySectorToEvents(self, sector, stage):
        sector_id = sector.sector_id
        sp = self.__getStageProcessor(sector_id, stage)
        mapping = {
            AnalysisStage.Preselection: sector.preselection,
            AnalysisStage.Selection: sector.selection,
            AnalysisStage.Categorization: sector.categories,
            AnalysisStage.Histogramming: sector.histograms,
            AnalysisStage.ObjectDefinition: sector.objects,
            AnalysisStage.Weights: sector.weights,
        }
        if stage in [AnalysisStage.Preselection]:
            events = self._sample_events[sector_id.sample_id]
        elif stage in [AnalysisStage.Selection, AnalysisStage.ObjectDefinition]:
            events = self._preselected_events[sector_id.sample_id]
        else:
            events = self._sector_events[sector_id]
        params = getParamsForSector(sector_id, self.dataset_repo, self.era_repo)
        for module in mapping[stage]:
            if sp:
                module(events, params, sp)
            else:
                module(events, params)

    def runStage(self, stage):
        for sector in self.sectors:
            self._applySectorToEvents(sector, stage)

    def applyPreselections(self):
        for sample_id, events in self._sample_events.items():
            self._preselected_events[
                sample_id
            ] = analyzer.selection_manager.maskPreselection(sample_id, events)

    def applyPreselections(self):
        for sample_id, events in self._sample_events.items():
            self._preselected_events[
                sample_id
            ] = self.selection_manager.maskPreselection(sample_id, events)

    def applySelection(self):
        self.selection_manager.addPreselectionMasks()
        for sector in self.sectors:
            sector_id = sector.sector_id
            events = self._preselected_events[sector_id.sample_id]
            self._sector_events[sector_id] = self.selection_manager.maskSector(
                sector_id, events
            )

    def populateCutflows(self):
        for sector in self.sectors:
            self._populateCutflow(sector.sector_id)


def runAnalysis(analyzer):
    if not analyzer.preprocessed_samples:
        logger.info(f"Analyzer does not already have preprocessed samples.")
        analyzer.preprocessDatasets()
    # When patching results, we may have sectors with no preprocessing object
    # since all the events have already been processed succesfully.
    analyzer.dropEmptySectors()
    logger.info(f"Running analysis with {len(analyzer.sectors)} sectors.")
    analyzer.loadEvents()

    mc_weights = analyzer.getTotalMcWeights()
    analyzer.runStage(AnalysisStage.Preselection)

    analyzer.applyPreselections()

    analyzer.runStage(AnalysisStage.ObjectDefinition)
    analyzer.runStage(AnalysisStage.Selection)

    analyzer.applySelection()

    analyzer.runStage(AnalysisStage.Weights)

    analyzer.populateCutflows()

    analyzer.runStage(AnalysisStage.Categorization)
    analyzer.runStage(AnalysisStage.Histogramming)

    logger.info(f"Finished lazy results construction")

    return AnalysisResult(
        description=analyzer.description,
        preprocessed_samples=analyzer.preprocessed_samples,
        processed_chunks={
            s: getProcessedChunks(r) for s, r in analyzer._sample_reports.items()
        },
        results=analyzer.results,
        total_mc_weights=mc_weights,
    )


def makeResultPatch(self, result, dataset_repo, era_repo):
    missing = result.getMissingPreprocessed()
    new_analyzer = Analyzer(result.description, dataset_repo, era_repo)
    new_analyzer.preprocessed_samples = missing


if __name__ == "__main__":
    import analyzer.modules
    from analyzer.datasets import DatasetRepo, EraRepo
    from analyzer.logging import setup_logging

    setup_logging(default_level=logging.INFO)

    d = yaml.safe_load(open("configurations/test_config.yaml", "r"))

    an = AnalysisDescription(**d)
    em = EraRepo()
    em.load("analyzer_resources/eras")
    dm = DatasetRepo()
    dm.load("analyzer_resources/datasets")

    a = Analyzer(an, dm, em)

    client = Client()
    res = runAnalysis(a)
    d = res.model_dump()
    x = dask.compute(d)[0]
    res = AnalysisResult(**x)

    # print(res)
    # print(res)
    import pickle as pkl

    with open("result.pkl", "wb") as f:
        pkl.dump(res, f)

    # for s, e in a._sector_events.items():
    #    e.visualize(filename=f"images/{s}.svg")
