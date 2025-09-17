from __future__ import annotations
import copy
import math

import itertools as it
import lz4.frame
import logging
from pathlib import Path
from rich import print
import pickle as pkl
from functools import cached_property
from rich.prompt import Confirm
from collections import defaultdict
from typing import Any
import gc
from contextlib import contextmanager

import analyzer.core.histograms as anh
import analyzer.core.region_analyzer as anr
import analyzer.core.selection as ans
from analyzer.core.dask_tools import reduceResults
import analyzer.core.specifiers as spec
import analyzer.datasets as ad
import pydantic as pyd
from analyzer.utils.querying import SimpleNestedPatternExpression, Pattern
from pydantic import (
    ConfigDict,
    computed_field,
    BaseModel,
    RootModel,
    model_validator,
    TypeAdapter,
    Field,
    AliasChoices,
    PrivateAttr,
)
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetParams, FileSet, SampleParams, SampleType
from analyzer.utils.structure_tools import accumulate, dictToFrozen, iadd
from .exceptions import ResultIntegrityError
from rich.progress import track

# from concurrent.futures import ProcessPoolExecutor, as_completed
from .common_types import Scalar


logger = logging.getLogger(__name__)


class BaseResult(pyd.BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    histograms: dict[str, anh.HistogramCollection] = pyd.Field(default_factory=dict)
    other_data: dict[str, Any] = pyd.Field(default_factory=dict)
    selection_flow: ans.SelectionFlow | None = None
    post_sel_weight_flow: dict[str, Scalar] | None = None
    pre_sel_weight_flow: dict[str, Scalar] | None = None

    _raw_selection_flow: ans.SelectionFlow | None = None

    @property
    def raw_selection_flow(self):
        if self._raw_selection_flow is None:
            return self.selection_flow
        return self._raw_selection_flow

    def includeOnly(self, histograms):
        self.histograms = {x: y for x, y in self.histograms.items() if x in histograms}

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def __iadd__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """

        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        new_post_weight = accumulate(
            [self.post_sel_weight_flow, other.post_sel_weight_flow]
        )
        new_pre_weight = accumulate(
            [self.post_sel_weight_flow, other.post_sel_weight_flow]
        )
        self.histograms = new_hists
        self.other_data = new_other
        self.post_sel_weight_flow = new_post_weight
        self.pre_sel_weight_flow = new_pre_weight
        self.selection_flow = self.selection_flow + other.selection_flow
        self._raw_selection_flow = self.raw_selection_flow + other.raw_selection_flow
        return self

    def scaled(self, scale):
        """
        Scale all data. Currently selection flow is left unscaled.
        """
        ret = BaseResult(
            histograms={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            selection_flow=self.selection_flow.scaled(scale),
            post_sel_weight_flow=(
                {x: y * scale for x, y in self.post_sel_weight_flow.items()}
                if self.post_sel_weight_flow is not None
                else None
            ),
            pre_sel_weight_flow=(
                {x: y * scale for x, y in self.pre_sel_weight_flow.items()}
                if self.pre_sel_weight_flow is not None
                else None
            ),
        )
        ret._raw_selection_flow = self.raw_selection_flow
        return ret


class SubSectorResult(pyd.BaseModel):
    region: anr.RegionAnalyzer
    base_result: BaseResult

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def __iadd__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.region != other.region:
            raise ResultIntegrityError("Cannot add different regions together.")
        self.base_result += other.base_result
        return self

    def includeOnly(self, histograms):
        self.base_result.includeOnly(histograms)

    def scaled(self, scale):
        return SubSectorResult(
            region=self.region, base_result=self.base_result.scaled(scale)
        )

    def getHist(self, name):
        return self.base_result.histograms[name]


class MultiSectorResult(pyd.RootModel):
    root: dict[str, SubSectorResult]

    def values(self):
        return self.root.values()

    def keys(self):
        return self.root.keys()

    def items(self):
        return self.root.items()

    def __getitem__(self, item):
        return self.root[item]

    def __iadd__(self, other):
        iadd(self.root, other.root)
        return self

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret


class SampleResultPeek(BaseModel):
    # nevents_ran: int = Field(alias="run_events")
    nevents_processed: int
    params: SampleParams

    _from_files: set[str] = PrivateAttr(default_factory=set)

    @property
    def processed_events(self):
        return self.nevents_processed

    def __iadd__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """

        if self.params != other.params:
            raise ResultIntegrityError(
                f"Error: Attempting to add incomaptible analysis peeks"
            )
        # self.nevents_ran += other.nevents_ran
        self.nevents_processed += other.nevents_processed
        self._from_files |= other._from_files
        return self

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret


class SampleResult(pyd.BaseModel):
    sample_id: ad.SampleId

    params: SampleParams
    results: MultiSectorResult

    file_set_ran: FileSet
    file_set_processed: FileSet

    @computed_field
    @cached_property
    def peek(self) -> SampleResultPeek:
        return SampleResultPeek(
            nevents_ran=self.file_set_ran.events,
            nevents_processed=self.processed_events,
            params=self.params,
        )

    @property
    def processed_events(self):
        return self.file_set_processed.events

    def includeOnly(self, histograms):
        for r in self.results.values():
            r.includeOnly(histograms)

    def compatible(self, other):
        # Compute the overlap of the two results, this must be empty
        fs = self.file_set_processed.intersect(other.file_set_processed)
        if not fs.empty:
            error = (
                f"Could not add analysis for results {self.sample_id} and {other.sample_id} because the file sets over which they successfully processed overlap."
                f"Overlapping files:\n{list(fs.files)}"
            )
            raise ResultIntegrityError(error)

        # We can only add results if the parameters are the same
        if self.params != other.params:
            raise ResultIntegrityError(
                f"Error: Attempting to merge incomaptible analysis results"
            )

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def __iadd__(self, other):
        self.compatible(other)
        self.file_set_ran += other.file_set_ran
        self.file_set_processed += other.file_set_processed
        self.results += other.results
        return self

    def scaled(self, scale, central_weight=None, rescale_weights=None):
        def getPreW(result):
            if rescale_weights is None:
                return 1
            return self.getReweightScale(result, central_weight, rescale_weights)

        return SampleResult(
            sample_id=self.sample_id,
            params=self.params,
            file_set_ran=self.file_set_ran,
            file_set_processed=self.file_set_processed,
            results={k: v.scaled(getPreW(v) * scale) for k, v in self.results.items()},
        )

    def getReweightScale(self, result, central, other):
        c = result.base_result.pre_sel_weight_flow.get(central)
        if c is None:
            return 1
        other_weights = [result.base_result.pre_sel_weight_flow.get(x) for x in other]
        individual = [(c / x) for x in other_weights if x is not None and x > 0]
        ret = math.prod(individual)
        return ret

    def scaleToPhysical(self, central_weight=None, rescale_weights=None):
        """
        Scale MC results to the correct value based on their lumi and cross section
        All results are scaled to remove effect of missing files/failed chunks.
        """
        if self.params.dataset.sample_type == SampleType.MC:
            scale = (
                self.params.dataset.lumi
                * self.params.x_sec
                / self.file_set_processed.events
            )
        else:
            scale = self.params.n_events / self.file_set_processed.events

        return self.scaled(
            scale, central_weight=central_weight, rescale_weights=rescale_weights
        )


class ResultFilePeek(RootModel):
    root: dict[ad.SampleId, SampleResultPeek]

    def values(self):
        return self.root.values()

    def keys(self):
        return self.root.keys()

    def items(self):
        return self.root.items()

    def __getitem__(self, item):
        return self.root[item]

    def __iadd__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        iadd(self.root, other.root)
        return self

    def addFromFile(self, f):
        for x in self.root.values():
            x._from_files.add(f)

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret


class MultiSampleResult(BaseModel):
    _MAGIC_ID: ClassVar[Literal[b"sstopresult"]] = b"sstopresult"
    _HEADER_SIZE: ClassVar[Literal[4]] = 4

    root: dict[ad.SampleId, SampleResult] | bytes

    # @model_validator(mode="after")
    # def cachePeek(self) -> MultiSampleResult:
    #     self.peek  # noqa
    #     return self

    @classmethod
    def peekFile(cls, f):
        maybe_magic = f.read(len(cls._MAGIC_ID))
        if maybe_magic == cls._MAGIC_ID:
            peek_size = int.from_bytes(f.read(cls._HEADER_SIZE), byteorder="big")
            ret = ResultFilePeek.model_validate(pkl.loads(f.read(peek_size)))
            return ret
        else:
            return cls.model_validate(pkl.loads(maybe_magic + f.read())).peek

    @classmethod
    def peekBytes(cls, data: bytes):
        if data[0 : len(cls._MAGIC_ID)] == cls._MAGIC_ID:
            header_value = data[
                len(cls._MAGIC_ID) : len(cls._MAGIC_ID) + cls._HEADER_SIZE
            ]
            peek_size = int.from_bytes(header_value, byteorder="big")
            peek = data[
                len(cls._MAGIC_ID)
                + cls._HEADER_SIZE : len(cls._MAGIC_ID)
                + cls._HEADER_SIZE
                + peek_size
            ]
            return ResultFilePeek.model_validate(pkl.loads(peek))
        else:
            return cls.model_validate(pkl.loads(data)).peek

    @classmethod
    def fromBytes(cls, data: bytes):
        if data[0 : len(cls._MAGIC_ID)] == cls._MAGIC_ID:
            header_value = data[
                len(cls._MAGIC_ID) : len(cls._MAGIC_ID) + cls._HEADER_SIZE
            ]
            peek_size = int.from_bytes(header_value, byteorder="big")
            peek = data[
                len(cls._MAGIC_ID)
                + cls._HEADER_SIZE : len(cls._MAGIC_ID)
                + cls._HEADER_SIZE
                + peek_size
            ]
            core_data = data[len(cls._MAGIC_ID) + cls._HEADER_SIZE + peek_size :]
            return cls.model_validate(pkl.loads(core_data))
        else:
            return cls.model_validate(pkl.loads(data))

    def toBytes(self, packed_mode=True) -> bytes:
        with self.ensureCompressed():
            if packed_mode:
                # peek = pkl.dumps(self.peek.model_dump())
                # core_data = pkl.dumps(self.model_dump())
                peek = pkl.dumps(self.peek.model_dump())
                core_data = pkl.dumps(self.model_dump())
                pl = len(peek)
                plb = (pl.bit_length() + 7) // 8
                if plb > self._HEADER_SIZE:
                    raise RuntimeError
                return (
                    self._MAGIC_ID
                    + pl.to_bytes(self._HEADER_SIZE, byteorder="big")
                    + peek
                    + core_data
                )
            else:
                return pkl.dumps(self.model_dump())

    @model_validator(mode="before")
    @classmethod
    def handleRoot(cls, data):
        if isinstance(data, dict) and "root" not in data:
            return dict(root=data)
        return data

    # def model_post_init(self, __context):
    #     self.decompress()

    @property
    def is_compressed(self):
        return isinstance(self.root, bytes)

    @computed_field
    @cached_property
    def peek(self) -> ResultFilePeek:
        with self.ensureDecompressed():
            return ResultFilePeek({k: v.peek for k, v in self.root.items()})

    @contextmanager
    def ensureDecompressed(self):
        is_compressed = self.is_compressed
        if is_compressed:
            self.decompress()
        yield
        if is_compressed:
            self.compress()

    @contextmanager
    def ensureCompressed(self):
        is_compressed = self.is_compressed
        if not is_compressed:
            self.compress()
        yield
        if not is_compressed:
            self.decompress()

    def compress(self):
        if not self.is_compressed:
            self.root = lz4.frame.compress(pkl.dumps(self.root))

    def decompress(self):
        if self.is_compressed:
            self.root = pkl.loads(lz4.frame.decompress(self.root))

    def includeOnly(self, histograms):
        with self.ensureDecompressed():
            for r in self.values():
                r.includeOnly(histograms)

    def values(self):
        with self.ensureDecompressed():
            return self.root.values()

    def keys(self):
        with self.ensureDecompressed():
            return self.root.keys()

    def items(self):
        with self.ensureDecompressed():
            return self.root.items()

    def __getitem__(self, item):
        with self.ensureDecompressed():
            return self.root[item]

    def __iadd__(self, other):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        with self.ensureDecompressed():
            iadd(self.root, other.root)
        return self

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret


class SectorResult(pyd.BaseModel):
    sector_params: spec.SectorParams
    result: BaseResult

    @property
    def params_dict(self):
        return self.sector_params.model_dump()

    @property
    def histograms(self):
        return self.result.histograms


class DatasetResult(pyd.BaseModel):
    dataset_params: DatasetParams
    results: dict[str, BaseResult]

    file_set_ran: FileSet
    file_set_processed: FileSet

    @staticmethod
    def fromSampleResult(sample_results):
        # All samples must belong to the same dataset
        if not len(set(x.sample_id.dataset_name for x in sample_results)) == 1:
            raise ResultIntegrityError()

        # Dataset is created by adding sample results
        return DatasetResult(
            dataset_params=sample_results[0].params.dataset,
            results=accumulate(
                [
                    {k: v.base_result for k, v in r.results.items()}
                    for r in sample_results
                ]
            ),
            file_set_ran=accumulate([x.file_set_ran for x in sample_results]),
            file_set_processed=accumulate(
                [x.file_set_processed for x in sample_results]
            ),
        )

    @property
    def sector_results(self):
        return [
            SectorResult(
                sector_params=spec.SectorParams(
                    dataset=self.dataset_params, region_name=rn
                ),
                result=result,
            )
            for rn, result in self.results.items()
        ]


def openAndLoad(path, include=None, decompress=False, peek_only=False):
    gc.disable()

    with open(path, "rb") as f:
        if peek_only:
            ret = MultiSampleResult.peekFile(f)
            ret.addFromFile(path)
        else:
            ret = MultiSampleResult.fromBytes(f.read())
            if decompress:
                ret.decompress()

            if include is not None:
                ret.includeOnly(include)

    gc.enable()

    return ret


def makeResultMap(paths):
    ret = defaultdict(list)
    for p in paths:
        with open(p, "rb") as f:
            gc.disable()
            data = pkl.load(f)
            gc.enable()
            for s in data.keys():
                ret[s].append(p)
    return ret


def mergeResult(results):
    return accumulate(results)


def combineResults(results):
    ret = {}
    for r in results:
        for k in list(r.keys()):
            if k not in ret:
                ret[k] = r[k]
            else:
                ret[k] += r[k]
    return ret


def loadSampleResultFromPaths(
    paths,
    include=None,
    parallel=CONFIG.DEFAULT_PARALLEL_PROCESSES,
    show_progress=False,
    decompress=False,
    peek_only=False,
):
    ret = {}

    if not parallel:
        paths = list(paths)
        iterator = track(
            paths,
            total=len(paths),
            transient=True,
            description="Loading Files",
            disable=not show_progress,
        )
        for p in iterator:
            try:
                r = openAndLoad(
                    p, include=include, decompress=decompress, peek_only=peek_only
                )
                for k in list(r.keys()):
                    if k not in ret:
                        ret[k] = r[k]
                    else:
                        ret[k] += r[k]
            except Exception:
                print(f"An error occurred while trying to load file {p}")
                raise
    else:
        from distributed import Client, LocalCluster, as_completed

        with LocalCluster(
            n_workers=parallel,
            processes=True,
            dashboard_address="localhost:8786",
            memory_limit="4GB",
        ) as cluster, Client(cluster) as client:
            futures = client.map(
                lambda x: openAndLoad(
                    x,
                    include,
                    decompress=decompress,
                    peek_only=peek_only,
                    include=include,
                ),
                paths,
            )
            final = reduceResults(
                client,
                accumulate,
                futures,
                reduction_factor=5,
                target_final_count=10,
            )
            iterator = track(
                as_completed(final),
                total=len(final),
                transient=True,
                description="Loading Files",
                disable=not show_progress,
            )
            for f in iterator:
                r = f.result()
                f.cancel()
                for k in list(r.keys()):
                    if k not in ret:
                        ret[k] = r[k]
                    else:
                        ret[k] += r[k]
    return ret


def gatherFilesByPattern(paths, fields):
    from analyzer.cli.quicklook import quicklookSample

    pattern = SimpleNestedPatternExpression.model_validate(
        {x: Pattern.Any() for x in fields}
    )

    group_paths = defaultdict(dict)

    params_dict = defaultdict(set)
    for p in track(paths, total=len(paths)):
        with open(p, "rb") as f:
            data = MultiSampleResult.peekFile(f)
        for sample in data.values():
            params = dictToFrozen(pattern.capture(sample.params))
            params_dict[params].add(str(p))
    return params_dict


def merge(paths, outdir, fields=None):
    from analyzer.cli.quicklook import quicklookSample

    fields = fields or ["dataset.name"]
    pattern = SimpleNestedPatternExpression.model_validate(
        {x: Pattern.Any() for x in fields}
    )

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    if any(outdir.iterdir()):
        ok = Confirm.ask(
            f"Output directory '{outdir}' is not empty. This may cause problems. Do you want to proceed?"
        )
        if not ok:
            print("Aborting")
            return

    group_paths = defaultdict(dict)

    params_dict = defaultdict(list)
    for p in track(paths, total=len(paths)):
        data = openAndLoad(p)
        for sample in data.values():
            params = dictToFrozen(pattern.capture(sample.params))
            params_dict[params].append(p)

    for params, paths in track(params_dict.items(), total=len(params_dict)):
        output_name = "_".join(x[1] for x in params)
        output = outdir / f"{output_name}.result"
        print(f'Saving "{output}"')
        loaded = loadSampleResultFromPaths(paths)
        if output.exists():
            raise ResultIntegrityError("Cannot overwrite when merging!")

        with open(output, "wb") as f:
            f.write(loaded.toBytes())


def makeDatasetResults(
    sample_results,
    drop_samples=None,
    drop_sample_fn=lambda x: False,
    include_samples_as_datasets=False,
):
    drop_samples = drop_samples or []
    scaled_sample_results = defaultdict(list)
    # Make datasets results by grouping samples for each dataset
    for result in sample_results.values():
        if result.sample_id in drop_samples or drop_sample_fn(result.sample_id):
            logger.info(f'Not including sample "{result.sample_id}"')
            continue
        if result.processed_events > 0:
            scaled_sample_results[result.sample_id.dataset_name].append(
                result.scaleToPhysical(
                    central_weight="unweighted",
                    rescale_weights=["pileup_sf", "btag_shape_sf", "puid_sf"],
                )
            )
    if not include_samples_as_datasets:
        return {
            x: DatasetResult.fromSampleResult(y)
            for x, y in scaled_sample_results.items()
        }
    else:
        ret = {}
        for x, y in scaled_sample_results.items():
            ret[x] = DatasetResult.fromSampleResult(y)
            for s in y:
                ds = DatasetResult.fromSampleResult([s])
                ds.dataset_params = copy.deepcopy(ds.dataset_params)
                ds.dataset_params.name = str(s.sample_id)
                ds.dataset_params.title = str(s.sample_id.sample_name)
                ret[s.sample_id] = ds
        return ret


def checkResult(
    paths, configuration=None, only_bad=False, peek_only=True, threshold=0.95
):
    from rich.console import Console
    from rich.style import Style
    from rich.table import Table

    console = Console()

    loaded = loadSampleResultFromPaths(
        paths, include=[], show_progress=True, parallel=None, peek_only=True
    )
    results = list(loaded.values())

    # from analyzer.utils.debugging import jumpIn
    # jumpIn(**locals())

    if configuration:
        # If a configuration is provided we also check for completely missing samples
        from analyzer.datasets import DatasetRepo
        from analyzer.core.configuration import loadDescription

        description = loadDescription(configuration)
        dataset_repo = DatasetRepo.getConfig()
        config_samples = set(description.getAllSamples(dataset_repo))
        missing_samples = sorted(list(config_samples - set(loaded)))

    table = Table(title="Missing Events")
    for x in ("Dataset Name", "Sample Name", "% Complete", "Processed", "Total"):
        table.add_column(x)

    for result in results:
        sample_id = result.params.sample_id
        exp = result.params.n_events
        val = result.processed_events
        diff = exp - val
        percent = round(val / exp * 100, 2)
        frac_done = val / exp
        done = (frac_done >= threshold) and (frac_done <= 1.0)

        if only_bad and done:
            continue

        table.add_row(
            sample_id.dataset_name,
            sample_id.sample_name,
            str(percent),
            f"{val}",
            f"{exp}",
            style=Style(color="green" if done else "red"),
        )
    if configuration:
        for sample_id in missing_samples:
            table.add_row(
                sample_id.dataset_name,
                sample_id.sample_name,
                "Missing",
                "Missing",
                "Missing",
                style=Style(color="red"),
            )
    console.print(table)


# def updateMeta(paths):
#     from analyzer.datasets import DatasetRepo, EraRepo
#
#     dataset_repo = DatasetRepo.getConfig()
#     era_repo = EraRepo.getConfig()
#     for path in paths:
#         result = openAndLoad(path)
#         for k in results.values():
#             sid = k.sample_id
#             p = dataset_repo[sid].params
#             p.dataset.populateEra(era_repo)
#             k.params = p
#         with open(path, "wb") as f:
#             pkl.dump({x: y.model_dump() for x, y in results.items()}, f)
