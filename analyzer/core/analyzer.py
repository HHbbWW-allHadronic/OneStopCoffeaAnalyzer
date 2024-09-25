# from __future__ import annotations
import concurrent.futures
import copy
import functools as ft
import itertools as it
import json
import logging
import operator as op
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from distributed.diagnostics import memray
from typing import Any, Optional, Union

import yaml

import awkward as ak
import dask_awkward as dak
import dask
import distributed
import pydantic as pyd
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
from .old_histograms import HistogramCollection, HistogramSpec, generateHistogramCollection
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
        track,
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
            n: self.preprocessed_samples[n].chunks.difference(self.processed_chunks[n])
            for n in self.preprocessed_samples
        }
        return ret

    def getMissingPreprocessed(self):
        logger.debug(f"Scanning for bad chunks")
        prepped = copy.deepcopy(self.preprocessed_samples)
        bad_chunks = self.getBadChunks()
        ret = {}
        for sid in prepped:
            logger.debug(f"Sample {sid} has {len(bad_chunks[sid])} bad chunks.")
            bad = bad_chunks[sid]
            if bad:
                ret[sid] = prepped[sid]
                ret[sid].limit_chunks = bad
        return ret

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

        return {x: ft.reduce(op.add, y) for x, y in scaled_sample_results.items()}


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
        self.client = distributed.client._get_global_client()

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

    def preprocessDatasets(
        self, file_kwargs_override=None, chunk_override=None, **kwargs
    ):
        chunk_size = self.description.general_config.get("preprocessing", {}).get(
            "chunk_size", 100000
        )
        if chunk_override:
            chunk_size = chunk_override
        print(f"CHUNK SIZE IS {chunk_size}")
        file_args = self.description.general_config["file_retrieval"]
        file_args.update(file_kwargs_override or {})
        logger.info(f"Preprocessing samples with chunk size {chunk_size}")
        logger.info(f"Preprocessing with file retrieval arguments:\n{file_args}")

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
            #known_base_form=maybe_base_form,
        ).events()
        return sample_id, events, report

    def loadEvents(self, **kwargs):
        file_args = self.description.general_config["file_retrieval"]
        logger.debug(f"Loading with file retrieval arguments:\n{file_args}")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            samples = set(x.sector_id.sample_id for x in self.sectors)
            for sample_id in samples:
                spre = self.preprocessed_samples[sample_id]
                ds = spre.getCoffeaDataset(self.dataset_repo, **file_args)
                if spre.form is not None:
                    maybe_base_form = ak.forms.from_json(decompress_form(spre.form))
                else:
                    maybe_base_form = None
                # f = executor.submit(
                #    Analyzer.__loadOne, sample_id, ds["files"], maybe_base_form
                # )
                f = Analyzer.__loadOne(sample_id, ds["files"], maybe_base_form)
                futures.append(f)

            # to_iter = concurrent.futures.as_completed(futures)
            to_iter = futures
            if CONFIG.PRETTY_MODE:
                to_iter = track(
                    to_iter, description="Loading Events", total=len(futures)
                )
            for future in to_iter:
                logger.debug(f"Done loading events for sample {sample_id}.")
                r = future
                # r = future.result()
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
            ] = self.selection_manager.maskPreselection(sample_id, events)

    def applySelection(self):
        self.selection_manager.addPreselectionMasks()
        for sector in self.sectors:
            sector_id = sector.sector_id
            events = self._preselected_events[sector_id.sample_id]
            e = self.selection_manager.maskSector(sector_id, events)
            self._sector_events[sector_id] = e

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


def patchPreprocessed(
    dataset_repo,
    preprocessed_samples,
    step_size=75000,
    file_retrieval_kwargs=None,
):
    frk = file_retrieval_kwargs or {}
    samples = {p.sample_id: p for p in copy.deepcopy(preprocessed_inputs)}
    missing_dict = {}
    for n, prepped in datasets.items():
        x = prepped.missingCoffeaDataset(**frk)
        logger.info(f"Found {len(x[n]['files'])} files missing from dataset {n}")
        if x[n]["files"]:
            missing_dict.update(x)
    new = preprocessRaw(missing_dict, step_size=step_size)
    for n, v in new.items():
        samples[n] = samples[n].addCoffeaChunks({n: v})
    return samples


def makeResultPatch(result, dataset_repo, era_repo):
    missing = result.getMissingPreprocessed()
    logger.info(f"Found {len(missing)} samples with missing chunks")
    if not missing:
        logger.info(f"No missing chunks, nothing to do")
        return
    new_analyzer = Analyzer(result.description, dataset_repo, era_repo)
    new_analyzer.preprocessed_samples = missing
    return new_analyzer


def preprocessAnalysis(input_path, output_path, chunk_size=None, file_kwargs=None):
    logger.info(f"Preprocessing analysis from file {input_path}")
    with open(input_path, "rb") as f:
        data = yaml.safe_load(f)
    an = AnalysisDescription(**data)
    logger.info(f"Loaded analysis description {an.name}")
    analyzer = Analyzer(an, DatasetRepo.getConfig(), EraRepo.getConfig())
    analyzer.preprocessDatasets(
        chunk_override=chunk_size, file_kwargs_override=file_kwargs
    )
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "wb") as f:
        pkl.dump(analyzer.preprocessed_samples, f)
    logger.info(f'Saved preprocessed samples to "{out}"')


def patchPreprocessedFile(input_path, output_path, chunk_size=100000, file_kwargs=None):
    with open(input_path, "rb") as f:
        data = yaml.safe_load(f)
    dm = DatasetRepo.getConfig()
    result = patchPreprocessed(
        dm, data, chunk_size=chunk_size, file_retrieval_kwargs=file_kwargs
    )
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "wb") as f:
        pkl.dump(result, f)


def patchAnalysisResult(input_path, output_path):
    logger.info(f'Patching analysis result from "{input_path}"')
    with open(input_path, "rb") as f:
        result = pkl.load(f)
    analyzer = makeResultPatch(result, DatasetRepo.getConfig(), EraRepo.getConfig())
    if not analyzer:
        return
    res = runAnalysis(analyzer)
    dumped = res.model_dump()
    computed = dask.compute(dumped)[0]
    final_result = AnalysisResult(**computed)
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "wb") as f:
        pkl.dump(final_result, f)


def runFromFile(input_path, output_path, preprocessed_input_path=None):
    with open(input_path, "rb") as config_file:
        data = yaml.safe_load(config_file)
    if preprocessed_input_path:
        with open(preprocessed_input_path, "rb") as prep_file:
            preprocessed_input = pkl.load(prep_file)
    else:
        preprocessed_input = {}
    an = AnalysisDescription(**data)
    analyzer = Analyzer(an, DatasetRepo.getConfig(), EraRepo.getConfig())
    analyzer.preprocessed_samples = preprocessed_input
    res = runAnalysis(analyzer)
    dumped = res.model_dump()
    #dumped["processed_chunks"] = {}
    grouped=True
    onebyone=True

    #c,r = dask.base.unpack_collections(dumped, traverse=True)
    #dsk = dask.base.collections_to_dsk(c, True)
    #print(dsk)

    if grouped:
        def groupBySample(data):
            def getS(x):
                if isinstance(x, SampleId):
                    return x.sample_name
                if isinstance(x, SectorId):
                    return x.sample_id.sample_name
                return x

            ret = {}
            for x, y in data.items():
                l = ret.setdefault(x, {})
                for k, v in y.items():
                    d = l.setdefault(getS(k), {})
                    d[k] = v

            keys = list(it.chain.from_iterable(x.keys() for x in ret.values()))
            return {key: {k: ret[k][key] for k in ret if key in ret[k]} for key in keys}

        p = {
            "processed_chunks": {x: y for x, y in res.processed_chunks.items()},
            "results": {x: y.model_dump() for x, y in res.results.items()},
            "total_mc_weights": {x: y for x, y in res.total_mc_weights.items()},
        }
        grouped = groupBySample(p)
        ret = {}
        p = 10000
        for sample, vals in grouped.items():
            p= p - 100
            logger.info(f'Submitting sample "{sample}" to cluster.')
            if onebyone:
                ret[sample] = dask.compute(vals)[0]
            else:
                ret[sample] = dask.persist(vals, priority=p)[0]

        logger.info(
            f"Done submitting all sampels to cluster, waiting for computation to finish..."
        )
        if onebyone:
            computed = ret
        else:
            computed = dask.compute(ret)[0]
        final = {}
        final["total_mc_weights"] = dict(
            it.chain.from_iterable(
                (x.get("total_mc_weights", {}).items() for x in computed.values())
            )
        )
        final["results"] = dict(
            it.chain.from_iterable((x["results"].items() for x in computed.values()))
        )

        final["processed_chunks"] = dict(
            it.chain.from_iterable(
                (x["processed_chunks"].items() for x in computed.values())
            )
        )
        final = {**dumped, **final}

    else:
        final = dask.compute(dumped)[0]


    final["processed_chunks"] = {  #
        x: getProcessedChunks(y) for x, y in final["processed_chunks"].items()
    }

    final_result = AnalysisResult(**final)
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "wb") as f:
        pkl.dump(final_result, f)


if __name__ == "__main__":
    import analyzer.modules
    from analyzer.datasets import DatasetRepo, EraRepo

    #logger.debug("TEST1")
    #logger.info("TEST2")
    #logger.error("TEST3")

    #import sys
    #sys.exit()

    d = yaml.safe_load(open("configurations/test_config.yaml", "r"))

    from analyzer.clients import createNewCluster

    if True:
        cluster = createNewCluster(
            "lpccondor",
            dict(
                n_workers=80,
                memory="4.0G",
                schedd_host=None,
                dashboard_host="localhost:12358",
                timeout=7200,
            ),
        )
        client = Client(cluster)
    else:
        client = Client(
            dashboard_address="localhost:12358",
            memory_limit="4GB",
            n_workers=2,
        )

    #preprocessAnalysis("configurations/test_config.yaml", "prepped.pkl")
    #with memray.memray_workers("memrayout/"):
    runFromFile(
        "configurations/test_config.yaml",
        "test_no_known.pkl",
        preprocessed_input_path="prepped.pkl",
    )

    print("DOOOOOOOOOOOOOOOOONNNNNNNNNNNNNNNNEEEEEEEEEEEEEEEEEEEEE")
    print("DOOOOOOOOOOOOOOOOONNNNNNNNNNNNNNNNEEEEEEEEEEEEEEEEEEEEE")
    print("DOOOOOOOOOOOOOOOOONNNNNNNNNNNNNNNNEEEEEEEEEEEEEEEEEEEEE")
    print("DOOOOOOOOOOOOOOOOONNNNNNNNNNNNNNNNEEEEEEEEEEEEEEEEEEEEE")
