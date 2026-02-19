from .executor import Executor


def getPremadeExcutors() -> dict[str, Executor]:
    from analyzer.core.executors.immediate_exec import ImmediateExecutor
    from analyzer.core.executors.dask_exec import LPCCondorDask, LocalDaskExecutor

    return {
        "imm-1000": ImmediateExecutor(chunk_size=1000),
        "imm-testing": ImmediateExecutor(chunk_size=1000, deepcopy_analyzer=False),
        "imm-10000": ImmediateExecutor(chunk_size=1000),
        "local-dask-4G-10000": LocalDaskExecutor(
            chunk_size=10000, min_workers=4, max_workers=4, timeout=None
        ),
        "local-dask-4G-100000": LocalDaskExecutor(
            chunk_size=100000,
            min_workers=4,
            max_workers=4,
        ),
        "local-dask-4G-400000": LocalDaskExecutor(
            chunk_size=400000,
            min_workers=4,
            max_workers=4,
        ),
        "lpc-dask-condor-4G-100000": LPCCondorDask(
            chunk_size=100000,
            min_workers=5,
            max_workers=250,
            container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12",
        ),
        "lpc-dask-condor-4G-80000": LPCCondorDask(
            chunk_size=80000,
            min_workers=5,
            max_workers=250,
            container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12",
        ),
        "lpc-dask-condor-4G-50000": LPCCondorDask(
            chunk_size=50000,
            min_workers=5,
            max_workers=250,
            container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12",
        ),
        "lpc-dask-condor-4G-400000": LPCCondorDask(
            chunk_size=400000,
            min_workers=5,
            max_workers=250,
            container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12",
        ),
        "lpc-dask-condor-6G-100000": LPCCondorDask(
            chunk_size=100000,
            min_workers=5,
            max_workers=250,
            worker_memory="6GB",
            container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12",
        ),
        "lpc-dask-condor-8G-100000": LPCCondorDask(
            chunk_size=100000,
            min_workers=5,
            max_workers=250,
            worker_memory="8GB",
            container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12",
        ),
    }
