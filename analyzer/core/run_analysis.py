import collections
import copy
import datetime
import importlib.resources as ir
import itertools as it
import logging
import pickle
import shutil
import tempfile
from pathlib import Path

import analyzer
import analyzer.core as ac
import analyzer.datasets as ds
import dask
from analyzer.file_utils import compressDirectory
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress, LocalCluster
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    SpinnerColumn,
)
from functools import partial
from .configuration import getConfiguration
from analyzer.file_utils import pickleWithParents

logger = logging.getLogger(__name__)


def makeIterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)


def createPackageArchive(zip_path=None, archive_type="zip"):
    """Compress the local analyzer package so that it can be used on worker nodes."""

    logger.info("Creating analyzer archive")
    if not zip_path:
        temp_path = Path(tempfile.gettempdir())
    else:
        temp_path = Path(zip_path)
    analyzer_path = Path(ir.files(analyzer))
    trimmed_path = temp_path / "trimmedanalyzer" / "analyzer"
    if trimmed_path.is_dir():
        shutil.rmtree(trimmed_path)
    # Copy analyzer directory, ignoring useless files
    temp_analyzer = shutil.copytree(
        analyzer_path,
        trimmed_path,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*~", "**/site-packages/*analyzer*"
        ),
    )
    package_path = shutil.make_archive(
        temp_path / "analyzer",
        archive_type,
        root_dir=trimmed_path.parent,
        base_dir="analyzer",
    )
    final_path = temp_path / f"analyzer.{archive_type}"
    logger.info(f"Created analyzer archive at {final_path}")
    return final_path


def transferAnalyzerToClient(client):
    analyzer_path = Path(ir.files(analyzer))
    compression_path = (
        Path(getConfiguration()["ENV_LOCAL_APPLICATION_DATA"]) / "compressed/"
    )
    p = str(
        compressDirectory(
            analyzer_path, "compressed", name=compression_path, archive_type="zip"
        )
    )
    logger.info(f"Transfer file {p} to workers.")
    client.upload_file(p)


def createClient(dask_schedd_address, transfer_analyzer=True):
    logger.info(f"Scheduler address is {dask_schedd_address}")
    if dask_schedd_address:
        logger.info(f"Connecting client to scheduler at {dask_schedd_address}")
        client = Client(dask_schedd_address)
        if transfer_analyzer:
            transferAnalyzerToClient(client)
    else:
        client = None
        logger.info("No scheduler address provided, running locally")
        client = Client(n_workers=1, memory_limit="8GB", threads_per_worker=1)
        logger.info("Created local client")
    return client


def createPreprocessedSamples(
    sample_manager,
    samples,
    step_size=150000,
    file_retrieval_kwargs=None,
):
    if file_retrieval_kwargs is None:
        file_retrieval_kwargs = {}
    samples = [sample_manager[x] for x in samples]
    all_sets = list(
        it.chain.from_iterable(
            makeIterable(x.getAnalyzerInput(**file_retrieval_kwargs)) for x in samples
        )
    )
    logger.info(f"Preprocessing {len(all_sets)} ")
    with Progress(TextColumn("{task.description}"), BarColumn()) as p:
        t = p.add_task("Preprocessing Files", total=None)
        dataset_preps = ac.preprocessBulk(
            all_sets, step_size=step_size, file_retrieval_kwargs=file_retrieval_kwargs
        )
    return dataset_preps


def patchPreprocessed(
    sample_manager,
    preprocessed_inputs,
    step_size=75000,
    file_retrieval_kwargs=None,
):
    if file_retrieval_kwargs is None:
        file_retrieval_kwargs = {}
    datasets = {
        p.dataset_input.dataset_name: p for p in copy.deepcopy(preprocessed_inputs)
    }
    missing_dict = {}
    for n, prepped in datasets.items():
        x = prepped.missingCoffeaDataset(**file_retrieval_kwargs)
        logger.info(f"Found {len(x[n]['files'])} files missing from dataset {n}")
        if x[n]["files"]:
            missing_dict.update(x)
    # logger.info(f"Processing the following missing data:\n{missing_dict}")
    with Progress(TextColumn("{task.description}"), BarColumn()) as p:
        t = p.add_task("Running", total=None)
        new = ac.inputs.preprocessRaw(missing_dict, step_size=step_size)
    for n, v in new.items():
        datasets[n] = datasets[n].addCoffeaChunks({n: v})
    return list(datasets.values())


def runModulesOnDatasets(
    modules,
    prepped_datasets,
    client=None,
    skim_save_path=None,
    file_retrieval_kwargs=None,
    include_default_modules=True,
    limit_samples=None,
    limit_files=None,
    sample_manager=None,
):
    import analyzer.modules

    if limit_samples and sample_manager is None:
        raise RuntimeError("If limiting samples must also provide sample manager")

    if limit_samples:
        samples = [sample_manager[x] for x in limit_samples]
        limited = list(
            it.chain.from_iterable(makeIterable(x.getAnalyzerInput()) for x in samples)
        )
        names = [x.dataset_name for x in limited]

        prepped_datasets = [x for x in prepped_datasets if x.dataset_name in names]

    config_path = Path(getConfiguration()["APPLICATION_DATA"])
    path = config_path / "argparse_cache" / f"modules.pkl"
    pickleWithParents(path, list(analyzer.core.org.modules))

    if file_retrieval_kwargs is None:
        file_retrieval_kwargs = {}
    file_retrieval_kwargs["modules"] = modules
    cache = {}
    analyzer = ac.Analyzer(modules, cache)
    futures = []
    with Progress(
        TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn()
    ) as p:
        t = p.add_task("Preparing Datasets", total=len(prepped_datasets))
        tasks = [
            (
                p.add_task(
                    x.dataset_name, total=len(analyzer.module_names), visible=False
                ),
                x,
            )
            for x in prepped_datasets
        ]
        for task, prepped in tasks:
            p.advance(t, 1)
            f = analyzer.getDatasetFutures(
                prepped,
                skim_save_path=skim_save_path,
                prog_bar_updater=partial(p.update, task),
                file_retrieval_kwargs=file_retrieval_kwargs,
                include_default_modules=include_default_modules,
                limit_files=limit_files,
            )
            futures.append(f)
    logger.info(f"Generated {len(futures)} analysis futures")
    with Progress(TextColumn("{task.description}"), BarColumn()) as p:
        t = p.add_task("Running Analysis", total=None)
        ret = ac.execute(futures, client)
    ret = ac.AnalysisResult(ret, modules, include_default_modules)
    return ret


def patchResult(result, **kwargs):
    modules = result.module_list
    missing_datasets = {
        k: v.getMissingDataset() for k, v in result.results.items() if v.getBadChunks()
    }
    ret = runModulesOnDatasets(
        modules,
        list(missing_datasets.values()),
        include_default_modules=result.use_default_modules,
        **kwargs,
    )
    return result.merge(ret)
