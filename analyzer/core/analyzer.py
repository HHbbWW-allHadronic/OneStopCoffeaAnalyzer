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
from typing import Any, Optional, Union

import yaml

import awkward as ak
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
from .histograms import HistogramCollection, HistogramSpec, generateHistogramCollection
from .preprocessed import SamplePreprocessed, preprocessBulk
from .sector import SubSector, getParamsForSubSector
from .selection import Cutflow, SelectionManager
from .specifiers import SectorParams, SubSectorId, SubSectorParams
from .weights import WeightManager

if CONFIG.PRETTY_MODE:
    from rich.progress import track


logger = logging.getLogger("analyzer.core")

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
        if self.weighted_sum is not None:
            new_ws = (
                (self.weighted_sum[0] + other.weighted_sum[0]),
                (self.weighted_sum[1] + other.weighted_sum[1]),
            )
        else:
            new_ws = None

        return SampleCutflow(
            cutflow=self.cutflow + other.cutflow,
            raw_passed=self.raw_passed + other.raw_passed,
            weighted_sum=new_ws,
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


class SubSectorResult(pyd.BaseModel):
    params: SubSectorParams
    histograms: dict[str, HistogramCollection] = pyd.Field(default_factory=dict)
    other_data: dict[str, Any] = pyd.Field(default_factory=dict)
    cutflow_data: Optional[SampleCutflow] = None

    def __add__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.params != other.params:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return SubSectorResult(
            params=self.params,
            histograms=new_hists,
            other_data=new_other,
            cutflow_data=self.cutflow_data + other.cutflow_data,
        )

    def scaled(self, scale):
        return SubSectorResult(
            params=self.params,
            histograms={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            cutflow_data=self.cutflow_data.scaled(scale),
        )


class SectorResult(pyd.BaseModel):
    sector_params: SectorParams
    sample_params: list[dict[str, Any]]
    histograms: dict[str, HistogramCollection]
    other_data: dict[str, Any]
    cutflow_data: Optional[SampleCutflow]

    @staticmethod
    def fromSubSectorResult(subsector_result):
        return SectorResult(
            sector_params=subsector_result.params.sector,
            sample_params=[subsector_result.params.sample],
            histograms=subsector_result.histograms,
            other_data=subsector_result.other_data,
            cutflow_data=subsector_result.cutflow_data,
        )

    def scaled(self, scale):
        return SectorResult(
            sector_params=self.sector_params,
            sample_params=[subsector_result.params.sample_params],
            histogrmams={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            cutflow_data=self.cutflow_data.scaled(scale),
        )

    def __add__(self, other):
        """Two subsector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if (
            self.dataset_params != other.dataset_params
            or self.era_params != other.era_params
        ):
            raise RuntimeError(f"Error: Attempting to merge incomaptible results")
        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return SectorResult(
            sector_params=self.sector_params,
            sample_params=self.sample_params + other.sample_params,
            histograms=new_hists,
            other_data=new_other,
            cutflow_data=self.cutflow_data + other.cutflow_data,
        )


class AnalysisResult(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)
    description: AnalysisDescription
    preprocessed_samples: dict[SampleId, SamplePreprocessed]
    processed_chunks: dict[SampleId, Any]
    results: dict[SubSectorId, SubSectorResult]
    total_mc_weights: dict[SampleId, Optional[Scalar]]

    @property
    def raw_events_processed(self):
        return {
            sample_id: sum(e - s for _, s, e in chunks)
            for sample_id, chunks in self.processed_chunks.items()
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

    def getResults(self, drop_samples=None):
        drop_samples = drop_samples or []
        scaled_sample_results = defaultdict(list)
        for subsector_id, result in self.results.items():
            if subsector_id.sample_id in drop_samples:
                continue
            k = (subsector_id.sample_id.dataset_name, subsector_id.region_name)

            if result.params.sector.dataset.sample_type == SampleType.MC:
                sample_info = result.params.sample
                scale = (
                    result.params.sector.dataset.lumi
                    * sample_info["x_sec"]
                    / self.total_mc_weights[subsector_id.sample_id]
                )
                result = result.scaled(scale)
            scaled_sample_results[k].append(SectorResult.fromSubSectorResult(result))

        return {x: ft.reduce(op.add, y) for x, y in scaled_sample_results.items()}

    @staticmethod
    def fromFile(path):
        with open(path, "rb") as f:
            data = pkl.load(f)
        return AnalysisResult(**data)


class SubSectorAnalyzer:
    class Selector:
        def __init__(self, parent, subsector_id, stage):
            self.parent = parent
            self.stage = stage
            self.subsector_id = subsector_id

        def add(self, name, mask, type="and"):
            return self.parent.addSelection(
                self.subsector_id, name, mask, type=type, stage=self.stage
            )

    class Weighter:
        def __init__(self, parent, subsector_id):
            self.parent = parent
            self.subsector_id = subsector_id

        def add(self, *args, **kwargs):
            return self.parent.addWeight(self.subsector_id, *args, **kwargs)

    class Categorizer:
        def __init__(self, parent, subsector_id):
            self.parent = parent
            self.subsector_id = subsector_id

        def add(self, *args, **kwargs):
            return self.parent.addCategory(self.subsector_id, *args, **kwargs)

    class Histogrammer:
        def __init__(self, parent, subsector_id):
            self.parent = parent
            self.subsector_id = subsector_id

        def addHistogram(self, *args, **kwargs):
            return self.parent.addHistogram(self.subsector_id, *args, **kwargs)

        def H(self, *args, **kwargs):
            return self.parent.makeHistogram(self.subsector_id, *args, **kwargs)


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
    _subsector_events: dict[SubSectorId, Any] = field(default_factory=dict)

    weight_manager: WeightManager = field(default_factory=WeightManager)
    selection_manager: SelectionManager = field(default_factory=SelectionManager)
    categories: defaultdict[str, dict[str, Category]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    subsectors: list[SubSector] = field(default_factory=list)

    results: dict[SubSectorId, SubSectorResult] = field(default_factory=dict)

    def __post_init__(self):
        self.subsectors = self._getSubSectors()
        self.__prefillResults()
        self.client = distributed.client._get_global_client()

    def __prefillResults(self):
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
        ]
        logger.debug(f"Dropping {len(self.subsectors) - len(new)} empty subsectors.")
        self.subsectors = new

    def preprocessDatasets(
        self, file_kwargs_override=None, chunk_override=None, **kwargs
    ):
        step_size = self.description.general_config.get("preprocessing", {}).get(
            "step_size", 100000
        )
        if chunk_override:
            step_size = chunk_override
        file_args = self.description.general_config["file_retrieval"]
        file_args.update(file_kwargs_override or {})
        logger.info(f"Preprocessing samples with chunk size {step_size}")
        logger.info(f"Preprocessing with file retrieval arguments:\n{file_args}")

        r = preprocessBulk(
            self.dataset_repo,
            set(x.subsector_id.sample_id for x in self.subsectors),
            step_size=step_size,
            file_retrieval_kwargs=file_args,
        )
        self.preprocessed_samples = {x.sample_id: x for x in r}

    @staticmethod
    def __loadOne(sample_id, files, maybe_base_form):
        logger.debug(f"Loading events for sample {sample_id}.")
        logger.debug(f"Loading files:\n {list(files)}.")
        num_tries = 0
        max_tries = 1
        # events, report = NanoEventsFactory.from_root(
        #     files,
        #     schemaclass=NanoAODSchema,
        #     uproot_options=dict(
        #         allow_read_errors_with_report=True,
        #         timeout=30,
        #     ),
        #     #known_base_form=maybe_base_form,
        # ).events()
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
                    # known_base_form=maybe_base_form,
                ).events()
                break
            except Exception as e:
                logger.warn(f"Error while loading file:\n{e}")
                if num_tries == max_tries:
                    raise
        return sample_id, events, report

    def loadEvents(self, **kwargs):
        file_args = self.description.general_config["file_retrieval"]
        logger.debug(f"Loading with file retrieval arguments:\n{file_args}")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            samples = set(x.subsector_id.sample_id for x in self.subsectors)
            for sample_id in samples:
                spre = self.preprocessed_samples[sample_id]
                ds = spre.getCoffeaDataset(self.dataset_repo, **file_args)
                if spre.form is not None:
                    maybe_base_form = ak.forms.from_json(decompress_form(spre.form))
                else:
                    maybe_base_form = None
                f = executor.submit(
                    Analyzer.__loadOne, sample_id, ds["files"], maybe_base_form
                )
                # f = Analyzer.__loadOne(sample_id, ds["files"], maybe_base_form)
                futures.append(f)

            to_iter = concurrent.futures.as_completed(futures)
            if CONFIG.PRETTY_MODE:
                to_iter = track(
                    to_iter, description="Loading Events", total=len(futures)
                )
            for future in to_iter:
                logger.debug(f"Done loading events for sample {sample_id}.")
                r = future.result()
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

        self.results[subsector_id].cutflow_data = SampleCutflow(
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
            AnalysisStage.Categorization: subsector.categories,
            AnalysisStage.Histogramming: subsector.histograms,
            AnalysisStage.ObjectDefinition: subsector.objects,
            AnalysisStage.Weights: subsector.weights,
        }
        if stage in [AnalysisStage.Preselection]:
            events = self._sample_events[subsector_id.sample_id]
        elif stage in [AnalysisStage.Selection, AnalysisStage.ObjectDefinition]:
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


def runAnalysis(analyzer):

    if not analyzer.preprocessed_samples:
        logger.info(f"Analyzer does not already have preprocessed samples.")
        analyzer.preprocessDatasets()
    # When patching results, we may have subsectors with no preprocessing object
    # since all the events have already been processed succesfully.
    analyzer.dropEmptySubSectors()
    logger.info(f"Running analysis with {len(analyzer.subsectors)} subsectors.")
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
    step_size=None,
    file_retrieval_kwargs=None,
):
    samples = {p.sample_id: p for p in copy.deepcopy(preprocessed_samples)}
    missing_dict = {}
    dr = DatasetRepo.getConfig()
    for n, prepped in samples.items():
        if file_retrieval_kwargs is None:
            frk = prepped.file_retrieval_kwargs or {}
        if step_size is None:
            step = prepped.step_size
        x = prepped.missingCoffeaDataset(dr, **frk)
        logger.info(f"Found {len(x['files'])} files missing from dataset {n}")
        if x["files"]:
            missing_dict.update(x)

    def k(x):
        return x.step_size

    g = it.groupby(missing_dict.items(), k)
    for ss, vals in g:
        new = preprocessRaw(dict(vals), step_size=ss)
        for n, v in new.items():
            samples[n] = samples[n].addCoffeaChunks({n: v})
    return list(samples.values())


def makeResultPatch(result, dataset_repo, era_repo):
    missing = result.getMissingPreprocessed()
    logger.info(f"Found {len(missing)} samples with missing chunks")
    if not missing:
        logger.info(f"No missing chunks, nothing to do")
        return
    new_analyzer = Analyzer(result.description, dataset_repo, era_repo)
    new_analyzer.preprocessed_samples = missing
    return new_analyzer


def preprocessAnalysis(input_path, output_path):
    logger.info(f"Preprocessing analysis from file {input_path}")
    with open(input_path, "rb") as f:
        data = yaml.safe_load(f)
    an = AnalysisDescription(**data)
    logger.info(f"Loaded analysis description {an.name}")
    dm = DatasetRepo.getConfig()
    samples = list(
        it.chain(*[[x.sample_id for x in dm[y].samples] for y in an.samples])
    )
    logger.info(f"Preprocessing {samples}")
    frk = an.general_config.get("file_retrieval", {})
    logger.info(f"Retrieval kwargs are {frk}")
    step_size = an.general_config.get("preprocessing", {}).get("step_size", 100000)

    result = preprocessBulk(dm, samples, step_size=step_size, file_retrieval_kwargs=frk)
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    to_dump = [x.model_dump() for x in result]
    with open(out, "wb") as f:
        pkl.dump(to_dump, f)
    logger.info(f'Saved preprocessed samples to "{out}"')


def patchPreprocessedFile(input_path, output_path, step_size=None, file_kwargs=None):
    with open(input_path, "rb") as f:
        data = pkl.load(f)
        data = [SamplePreprocessed(**x) for x in data]

    dm = DatasetRepo.getConfig()
    result = patchPreprocessed(
        dm, data, step_size=step_size, file_retrieval_kwargs=file_kwargs
    )
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    to_dump = [x.model_dump() for x in result]
    with open(out, "wb") as f:
        pkl.dump(to_dump, f)


def patchAnalysisResult(input_path, output_path):
    logger.info(f'Patching analysis result from "{input_path}"')
    with open(input_path, "rb") as f:
        result = pkl.load(f)
        result = AnalysisResult(**result)
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
        pkl.dump(final_result.model_dump(), f)


def runFromFile(input_path, output_path, preprocessed_input_path=None):

    logger.info(distributed.client._get_global_client())
    with open(input_path, "rb") as config_file:
        data = yaml.safe_load(config_file)
    if preprocessed_input_path:
        with open(preprocessed_input_path, "rb") as prep_file:
            preprocessed_input = pkl.load(prep_file)
            preprocessed_input = [SamplePreprocessed(**x) for x in preprocessed_input]
            preprocessed_input = {x.sample_id: x for x in preprocessed_input}
    else:
        preprocessed_input = {}
    an = AnalysisDescription(**data)
    analyzer = Analyzer(an, DatasetRepo.getConfig(), EraRepo.getConfig())
    analyzer.preprocessed_samples = preprocessed_input
    res = runAnalysis(analyzer)
    dumped = res.model_dump()

    def groupBySample(data):
        def getS(x):
            if isinstance(x, SampleId):
                return x  # .sample_name
            if isinstance(x, SubSectorId):
                return x.sample_id  # .sample_name
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
    l = list(grouped.items())
    l = sorted(l, key=lambda x: len(analyzer.preprocessed_samples[x[0]].chunks))
    min_chunks_per_submit = 50
    current_list = []
    current = 0

    ret = {}
    for sample, val in l:
        current += len(analyzer.preprocessed_samples[sample].chunks)
        current_list.append((sample, val))
        if current >= min_chunks_per_submit:
            logger.info(f"Processing samples: {[x for x,_ in current_list]}")
            try:
                portion_computed = dask.compute(current_list)[0]
                for sample, res in portion_computed:
                    logger.info(f'Adding sample "{sample}" to result.')
                    ret[sample] = res
            except Exception as e:
                logger.error(
                    f"An error occurred that caused an exception dask.compute. "
                    f"This caused {len(current_list)} samples ({current} chunks) to be discarded from computation."
                    f"The analyzer will continue processing other samples, but will need to patched. "
                    f"This was caused by the following exception:\n{e}."
                )
            current = 0
            current_list = []
    if current_list:
        logger.info(f"Processing samples: {[x for x,_ in current_list]}")
        try:
            portion_computed = dask.compute(current_list)[0]
            for sample, res in portion_computed:
                logger.info(f'Adding sample "{sample}" to result.')
                ret[sample] = res
        except Exception as e:
            logger.error(
                f"An error occurred that caused an exception dask.compute. "
                f"This caused {len(current_list)} samples ({current} chunks) to be discarded from computation."
                f"The analyzer will continue processing other samples, but will need to patched. "
                f"This was caused by the following exception:\n{e}."
            )
    logger.info(f"Done processing all samples.")

    computed = ret
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


    final["processed_chunks"] = {  #
        x: getProcessedChunks(y) for x, y in final["processed_chunks"].items()
    }

    final = {**dumped, **final}


    final_result = AnalysisResult(**final)

    out = Path(output_path)

    out.parent.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving results to {out}")

    try:
        with open(out, "wb") as f:
            pkl.dump(final_result.model_dump(), f)
    except Exception as e:
        logger.error(f"An error occurred while attempting to save the results:\n {e}")
        logger.error(final_result.model_dump())
        raise e
