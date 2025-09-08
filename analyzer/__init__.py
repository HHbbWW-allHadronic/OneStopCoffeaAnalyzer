import warnings
from analyzer.configuration import CONFIG


if CONFIG.SUPPRESS_WARNINGS:
    warnings.filterwarnings("ignore", module="coffea.*")
    warnings.filterwarnings("ignore", module="XRootD.*")
