import datetime
import logging
import os
import shutil
import time
from pathlib import Path

import analyzer
import dask
import yaml
from analyzer.utils.file_tools import compressDirectory
from distributed import LocalCluster
from rich.progress import Progress

from .configuration import CONFIG

try:
    from lpcjobqueue import LPCCondorCluster
    from lpcjobqueue.schedd import SCHEDD

    LPCQUEUE_AVAILABLE = True
except ImportError as e:
    LPCQUEUE_AVAILABLE = False

logger = logging.getLogger(__name__)


def createLPCCondorCluster(
    max_workers=10,
    memory="2GB",
    dashboard_address="localhost:8787",
    schedd_address="localhost:12358",
    extra_files=None,
    timeout="7200",
    temporary_path=".temporary",
    **kwargs,
):
    """Create a new dask cluster for use with LPC condor."""
    if not LPCQUEUE_AVAILABLE:
        raise NotImplemented("LPC Condor can only be used at the LPC.")
    extra_files = extra_files or []

    apptainer_container = "/".join(Path(os.environ["APPTAINER_CONTAINER"]).parts[-2:])

    logpath = Path("/uscmst1b_scratch/lpc1/3DayLifetime/") / os.getlogin() / "dask_logs"
    logpath.mkdir(exist_ok=True, parents=True)

    base = Path(os.environ.get("APPTAINER_WORKING_DIR", ".")).resolve()
    venv = Path(os.environ.get("VIRTUAL_ENV"))
    x509 = Path(os.environ.get("X509_USER_PROXY")).absolute()
    logger.info("Deleting old dask logs")
    base_log_path = Path("/uscmst1b_scratch/lpc1/3DayLifetime/") / os.getlogin()
    shutil.rmtree(base_log_path / "dask_logs")
    for p in base_log_path.glob("tmp*"):
        shutil.rmtree(p)

    # Compress environment to transport to the condor workers
    compressed_env = Path(CONFIG.APPLICATION_DATA) / "compressed" / "environment.tar.gz"
    analyzer_compressed = (
        Path(CONFIG.APPLICATION_DATA) / "compressed" / "analyzer.tar.gz"
    )
    if not compressed_env.exists():
        compressDirectory(
            input_dir=".application_data/venv",
            root_dir="/srv/",
            output=compressed_env,
            archive_type="gztar",
        )
    compressDirectory(
        input_dir=Path(analyzer.__file__).parent.relative_to("/srv/"),
        root_dir="/srv/",
        output=analyzer_compressed,
        archive_type="gztar",
    )

    transfer_input_files = ["setup.sh", compressed_env, analyzer_compressed]

    if extra_files:
        extra_compressed = (
            Path(CONFIG.APPLICATION_DATA) / "compressed" / "extra_files.tar.gz"
        )
        transfer_input_files.append(extra_compressed)
        temp = Path(temporary_path)
        extra_files_path = temp / "extra_files/" 
        extra_files_path.mkdir(exist_ok=True, parents=True)
        for i in extra_files:
            src = Path(i)
            print(src)
            shutil.copytree(src, extra_files_path / i)

        compressDirectory(
            input_dir="",
            root_dir=extra_files_path,
            output=extra_compressed,
            archive_type="gztar",
        )


    # transfer_input_files = [
    #    str(base / "setup.sh"),
    #    str(base / str(compressed_env)),
    # ]
    if x509:
        transfer_input_files.append(x509)
    kwargs = {}
    kwargs["worker_extra_args"] = [
        *dask.config.get("jobqueue.lpccondor.worker_extra_args"),
        # "--preload",
        # "lpcjobqueue.patch",
    ]
    kwargs["job_extra_directives"] = {"+MaxRuntime": timeout}
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
            # address=schedd_host,
            dashboard_address=":12358"
        ),
        **kwargs,
    )
    # cluster.scale(jobs=max_workers)
    # print(cluster)
    cluster.adapt(minimum=1, maximum=max_workers)

    return cluster


def createLocalCluster(
    max_workers=10,
    memory="2GB",
    dashboard_address="localhost:8787",
    schedd_address="localhost:12358",
    **kwargs,
):
    cluster = LocalCluster(
        dashboard_address=dashboard_address,
        memory_limit=memory,
        n_workers=max_workers,
        scheduler_kwargs={"host": schedd_address},
    )
    return cluster


cluster_factory = dict(
    local=createLocalCluster,
    lpccondor=createLPCCondorCluster,
)


def createNewCluster(cluster_type, config):
    """Creates a general new cluster of a certain given a configuration."""

    with open(CONFIG.DASK_CONFIG_PATH) as f:
        defaults = yaml.safe_load(f)
        dask.config.update(dask.config.config, defaults, priority="new")
        cluster = cluster_factory[cluster_type](**config)
    return cluster


def cleanup(p):
    if p.is_alive():
        logger.info("Cleaning up process")
        p.terminate()
        p.join()


def runNewCluster(cluster_type, config):
    cluster = createNewCluster(cluster_type, config)
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=config["timeout"])
    with Progress() as progress:
        running = progress.add_task("[red]Remaining Time...", total=(config["timeout"]))
        while True:
            time.sleep(1)
            now = datetime.datetime.now()
            progress.update(running, advance=1)
            if now > end_time:
                break
