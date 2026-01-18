import warnings
from analyzer.configuration import CONFIG


if CONFIG.general.suppress_coffea_warnings:
    warnings.filterwarnings("ignore", module="coffea.*")
    warnings.filterwarnings("ignore", module="analyzer.coffea_patches.*")
if CONFIG.general.suppress_xrootd_warnings:
    warnings.filterwarnings("ignore", module="XRootD.*")
