import copy
import logging
from pathlib import Path
import pickle as pkl
from collections import defaultdict
from typing import Any

import analyzer.core.histograms as anh
import analyzer.core.region_analyzer as anr
import analyzer.core.selection as ans
import analyzer.core.specifiers as spec
import analyzer.datasets as ad
import pydantic as pyd
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetParams, FileSet, SampleParams, SampleType
from analyzer.utils.structure_tools import accumulate
from rich.progress import track
from concurrent.futures import ProcessPoolExecutor, as_completed


logger = logging.getLogger(__name__)


class BaseResult(pyd.BaseModel):
    histograms: dict[str, anh.HistogramCollection] = pyd.Field(default_factory=dict)
    other_data: dict[str, Any] = pyd.Field(default_factory=dict)
    selection_flow: ans.SelectionFlow | None = None

    def includeOnly(self, histograms):
        self.histograms = {x: y for x, y in self.histograms.items() if x in histograms}

    def __add__(self, other: "SubSectorResult"):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """

        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        return BaseResult(
            histograms=new_hists,
            other_data=new_other,
            selection_flow=self.selection_flow + other.selection_flow,
        )

    def __iadd__(self, other: "SubSectorResult"):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """

        new_hists = accumulate([self.histograms, other.histograms])
        new_other = accumulate([self.other_data, other.other_data])
        self.histograms = new_hists
        self.other_data = new_other
        self.selection_flow = self.selection_flow + other.selection_flow
        return self

    def scaled(self, scale):
        """
        Scale all data. Currently selection flow is left unscaled.
        """
        return BaseResult(
            histograms={x: y.scaled(scale) for x, y in self.histograms.items()},
            other_data=self.other_data,
            selection_flow=self.selection_flow,  # .scaled(scale),
        )


class SubSectorResult(pyd.BaseModel):
    region: anr.RegionAnalyzer
    base_result: BaseResult

    def __add__(self, other: "SubSectorResult"):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.region != other.region:
            raise RuntimeError("Cannot add different regions together.")
        return SubSectorResult(
            region=self.region, base_result=self.base_result + other.base_result
        )

    def __iadd__(self, other: "SubSectorResult"):
        """Two SubSector results may be added if they have the same parameters
        We simply sum the histograms and cutflow data.
        """
        if self.region != other.region:
            raise RuntimeError("Cannot add different regions together.")
        self.base_result += other.base_result
        return self

    def includeOnly(self, histograms):
        self.base_result.includeOnly(histograms)

    def scaled(self, scale):
        return SubSectorResult(
            region=self.region, base_result=self.base_result.scaled(scale)
        )


class SampleResult(pyd.BaseModel):
    sample_id: ad.SampleId

    params: SampleParams
    results: dict[str, SubSectorResult]

    file_set_ran: FileSet
    file_set_processed: FileSet

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
                f"Could not add analysis results because the file sets over which they successfully processed overlap."
                f"Overlapping files:\n{fs}"
            )
            raise RuntimeError(error)

        # We can only add results if the parameters are the same
        if self.params != other.params:
            raise RuntimeError(
                f"Error: Attempting to merge incomaptible analysis results"
            )

    def __add__(self, other):
        self.compatible(other)

        return SampleResult(
            sample_id=self.sample_id,
            params=self.params,
            file_set_ran=self.file_set_ran + other.file_set_ran,
            file_set_processed=self.file_set_processed + other.file_set_processed,
            results=accumulate([self.results, other.results]),
        )

    def __iadd__(self, other):
        self.compatible(other)
        self.file_set_ran += other.file_set_ran
        self.file_set_processed += other.file_set_processed
        self.results = accumulate([self.results, other.results])
        return self

    def scaled(self, scale):
        return SampleResult(
            sample_id=self.sample_id,
            params=self.params,
            file_set_ran=self.file_set_ran,
            file_set_processed=self.file_set_processed,
            results={k: v.scaled(scale) for k, v in self.results.items()},
        )

    def scaleToPhysical(self):
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
        return self.scaled(scale)


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
        # All samples must belong fo the same dataset
        if not len(set(x.sample_id.dataset_name for x in sample_results)) == 1:
            raise RuntimeError()

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


results_adapter = pyd.TypeAdapter(dict[ad.SampleId, SampleResult])
subsector_adapter = pyd.TypeAdapter(dict[str, SubSectorResult])


def loadResults(obj):
    return results_adapter.validate_python(obj)


def openAndLoad(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return loadResults(data)


def loadSampleResultFromPaths(
    paths, include=None, parallel=CONFIG.DEFAULT_PARALLEL_PROCESSES
):
    ret = {}

    if not parallel:
        iterator = track(
            paths,
            total=len(paths),
            transient=True,
            description="Loading Files",
            disable=not CONFIG.PRETTY_MODE,
        )
        for p in iterator:
            with open(p, "rb") as f:
                data = pkl.load(f)
                r = loadResults(data)
                for k in list(r.keys()):
                    if include is not None:
                        r[k].includeOnly(include)
                    if k not in ret:
                        ret[k] = r[k]
                    else:
                        ret[k] += r[k]
                    del r[k]
    else:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(openAndLoad, path) for path in paths]
            # futures = executor.map(openAndLoad, paths)
            iterator = track(
                as_completed(futures),
                total=len(paths),
                transient=True,
                description="Loading Files",
                disable=not CONFIG.PRETTY_MODE,
            )
            for f in iterator:
                r = f.result()
                for k in list(r.keys()):
                    if include is not None:
                        r[k].includeOnly(include)
                    if k not in ret:
                        ret[k] = r[k]
                    else:
                        ret[k] += r[k]
                    del r[k]
    return ret


def merge(paths, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    results = loadSampleResultFromPaths(paths)
    for sample_id, result in results.items():
        output = outdir / f"{sample_id}.pkl"
        print(f'Saving sample {sample_id} to "{output}"')
        if output.exists():
            raise RuntimeError("Cannot overwrite when merging!")

        with open(output, "wb") as f:
            pkl.dump({sample_id: result.model_dump()}, f)


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
            logger.warn(f'Not including sample "{result.sample_id}"')
            continue
        if result.processed_events > 0:
            scaled_sample_results[result.sample_id.dataset_name].append(
                result.scaleToPhysical()
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
                print(s.sample_id)
        return ret


def checkResult(paths, configuration=None):
    from rich.console import Console
    from rich.style import Style
    from rich.table import Table

    console = Console()

    loaded = loadSampleResultFromPaths(paths, include=[])
    results = list(loaded.values())

    # from analyzer.utils.debugging import jumpIn
    # jumpIn(**locals())

    if configuration:
        # If a configuration is provided we also check for completely missing samples
        from analyzer.datasets import DatasetRepo
        from analyzer.core.configuration import loadDescription

        description = loadDescription(configuration)
        dataset_repo = DatasetRepo.getConfig()
        config_samples = set(
            s.sample_id
            for n in description.samples
            for x in dataset_repo.getRegex(n)
            for s in x.samples
        )
        missing_samples = sorted(list(config_samples - set(loaded)))

    table = Table(title="Missing Events")
    for x in ("Dataset Name", "Sample Name", "% Complete", "Processed", "Total"):
        table.add_column(x)

    for result in results:
        sample_id = result.params.sample_id
        exp = result.params.n_events
        val = result.processed_events
        diff = exp - val
        done = diff == 0
        percent = round(val / exp * 100, 2)

        table.add_row(
            sample_id.dataset_name,
            sample_id.sample_name,
            str(percent),
            f"{val}",
            f"{exp}",
            style=Style(color="green" if done else "red"),
        )
        # print(f"{sample_id} is missing {diff} events")
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


def updateMeta(paths):
    from analyzer.datasets import DatasetRepo, EraRepo

    dataset_repo = DatasetRepo.getConfig()
    era_repo = EraRepo.getConfig()
    for path in paths:
        with open(path, "rb") as f:
            data = pkl.load(f)
        results = loadResults(data)
        for k in results.values():
            sid = k.sample_id
            p = dataset_repo[sid].params
            p.dataset.populateEra(era_repo)
            k.params = p
        with open(path, "wb") as f:
            pkl.dump({x: y.model_dump() for x, y in results.items()}, f)
