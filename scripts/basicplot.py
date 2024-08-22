import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

sig = [
    f"signal_312_{p}"
    for p in (
        "1000_400",
        "1000_600",
        "1000_900",
        "1200_400",
        "1200_600",
        "1200_1100",
        "1400_400",
        "1400_600",
        "1400_1300",
        "1500_400",
        "1500_600",
        "1500_900",
        "1500_1400",
        "2000_400",
        "2000_600",
        "2000_900",
        "2000_1400",
        "2000_1900",
        "1000_700",
        "1000_800",
        "1200_700",
        "1200_800",
        "1200_900",
        "1200_1000",
        "1500_1000",
        "1500_1100",
        "1500_1200",
        "1500_1300",
        "1500_1350",
        "1500_1450",
        "2000_1200",
        "2000_1300",
        "2000_1500",
        "2000_1600",
        "2000_1700",
    )
]


backgrounds = ["QCDInclusive2018"]
plotter = Plotter("results/histograms/2018mc.pkl", "plots", default_backgrounds=backgrounds)


toplot = [
    "h_njet",
    "HT",
    "pt_1",
    "pt_2",
    "m14_m",
    "m13_m",
    "m24_vs_m14",
    "m13_vs_m14",
]

for p in toplot:
    plotter(p, [], normalize=False, scale="log")
