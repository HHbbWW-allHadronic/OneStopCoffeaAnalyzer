import atexit
import importlib.resources
import logging
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import dask
import yaml
from distributed import Client, LocalCluster, TimeoutError
from lpcjobqueue import LPCCondorCluster
from lpcjobqueue.schedd import SCHEDD

import analyzer.resources
from analyzer.file_utils import compressDirectory

logger = logging.getLogger(__name__)


def createLPCCondorCluster(configuration):
    apptainer_container = "/".join(Path(os.environ["APPTAINER_CONTAINER"]).parts[-2:])
    print(apptainer_container)
    workers = configuration["n_workers"]
    memory = configuration["memory"]
    schedd_host = configuration["schedd_host"]
    dash_host = configuration["dashboard_host"]
    logpath = Path("/uscmst1b_scratch/lpc1/3DayLifetime/") / os.getlogin() / "dask_logs"
    logpath.mkdir(exist_ok=True, parents=True)

    base = Path(os.environ.get("APPTAINER_WORKING_DIR", ".")).resolve()
    venv = Path(os.environ.get("VIRTUAL_ENV"))

    logger.info("Deleting old dask logs")
    base_log_path = Path("/uscmst1b_scratch/lpc1/3DayLifetime/ckapsiak/")
    shutil.rmtree(base_log_path / "dask_logs")
    for p in base_log_path.glob("tmp*"):
        shutil.rmtree(p)

    compressed_env = Path("compressed") / "environment.tar.gz"
    if not compressed_env.exists():
        compressDirectory(venv, compressed_env.parent)

    transfer_input_files = [
        str(base / "setup.sh"),
        str(base / str(compressed_env)),
    ]
    # transfer_input_files = ["setup.sh", compressed_env]
    # transfer_input_files = ["setup.sh"]
    kwargs = {}
    kwargs["worker_extra_args"] = [
        *dask.config.get("jobqueue.lpccondor.worker_extra_args"),
        # "--preload",
        # "lpcjobqueue.patch",
    ]

    kwargs["python"] = f"{venv}/bin/python"

    logger.info(f"Transfering input files: \n{transfer_input_files}")
    s = SCHEDD()
    # print(s)

    cluster = LPCCondorCluster(
        ship_env=False,
        image=apptainer_container,
        memory=memory,
        transfer_input_files=transfer_input_files,
        log_directory=logpath,
        scheduler_options=dict(
            host=schedd_host,
            dashboard_address=dash_host,
        ),
        **kwargs,
    )
    cluster.scale(workers)
    # print(cluster)
    # cluster.adapt(minimum=workers, maximum=workers)

    return cluster


def createLocalCluster(configuration):
    workers = configuration["n_workers"]
    memory = configuration["memory"]
    schedd_host = configuration["schedd_host"]
    dash_host = configuration["dashboard_host"]
    cluster = LocalCluster(
        dashboard_address=dash_host,
        memory_limit=memory,
        n_workers=workers,
        scheduler_kwargs={"host": schedd_host},
    )
    return cluster


cluster_factory = dict(
    local=createLocalCluster,
    lpccondor=createLPCCondorCluster,
)


def createNewCluster(cluster_type, config):
    with importlib.resources.as_file(
        importlib.resources.files(analyzer.resources)
    ) as f:
        dask_cfg_path = Path(f) / "dask_config.yaml"
    with open(dask_cfg_path) as f:
        defaults = yaml.safe_load(f)
        dask.config.update(dask.config.config, defaults, priority="new")
        cluster = cluster_factory[cluster_type](config)
        print(cluster)
        print(cluster.dashboard_link)
    return cluster


def cleanup(p):
    if p.is_alive():
        logger.info("Cleaning up process")
        p.terminate()
        p.join()


def runNewCluster(cluster_type, config):
    cluster = createNewCluster(cluster_type, config)
    time.sleep(config["timeout"])
