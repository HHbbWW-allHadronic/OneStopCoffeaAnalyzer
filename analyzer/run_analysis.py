import importlib.resources as ir
import itertools as it
import logging
import shutil
import tempfile
from pathlib import Path

import analyzer
import analyzer.core as ac
import analyzer.datasets as ds
import dask
from analyzer.file_utils import compressDirectory
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from rich.console import Console

logger = logging.getLogger(__name__)


def createPackageArchive(zip_path=None, archive_type="zip"):
    logger.info("Creating analyzer archive")
    if not zip_path:
        temp_path = Path(tempfile.gettempdir())
    else:
        temp_path = Path(zip_path)
    analyzer_path = Path(ir.files(analyzer))
    trimmed_path = temp_path / "trimmedanalyzer" / "analyzer"
    if trimmed_path.is_dir():
        shutil.rmtree(trimmed_path)
    temp_analyzer = shutil.copytree(
        analyzer_path,
        trimmed_path,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*~"),
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
    p = str(compressDirectory(analyzer_path, name="analyzer", archive_type="zip"))
    logger.info(f"Transfer file {p} to workers.")
    client.upload_file(p)


def runAnalysisOnSamples(
    modules,
    samples,
    sample_manager,
    dask_schedd_address=None,
    dataset_directory="datasets",
    step_size=75000,
    delayed=True,
):
    import analyzer.modules

    cache = {}
    if dask_schedd_address:
        logger.info(f"Connecting client to scheduler at {dask_schedd_address}")
        client = Client(dask_schedd_address)
        transferAnalyzerToClient(client)
    else:
        client = None
        logger.info("No scheduler address provided, running locally")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")
    logger.info(f"Creating analyzer using {len(modules)} modules")
    analyzer = ac.Analyzer(modules, cache)
    samples = [sample_manager[x] for x in samples]
    all_sets = list(it.chain.from_iterable(x.getAnalyzerInput() for x in samples))
    logger.info(f"Preprocessing {len(all_sets)} ")
    with ProgressBar():
        dataset_preps = ac.preprocessBulk(all_sets, step_size=step_size)

    # x = dict((d.dataset_input.dataset_name, d.coffea_dataset_split) for d in dataset_preps)
    # import json

    # print(x)
    # json.dump(x, open("out.json", "w"))
    # return None

    logger.info(f"Preprocessed data in to {len(dataset_preps)} set")
    if delayed:
        futures = [analyzer.getDatasetFutures(x) for x in dataset_preps]
        logger.info(f"Generated {len(futures)} analysis futures")
        with ProgressBar():
            ret = ac.execute(futures, client)
    else:
        results = [analyzer.getDatasetFutures(x, delayed=False) for x in dataset_preps]
        ret = {x.getName(): x for x in results}
    print(ret)

    ret = ac.AnalysisResult(ret)
    return ret
