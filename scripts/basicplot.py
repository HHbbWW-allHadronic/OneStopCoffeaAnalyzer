import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = []
sig = [f"signal_312_{p}" for p in ("1500_400_MG",)]


plotter = Plotter("results/mgsamples.pkl", "fordevin", default_backgrounds=backgrounds)


toplot = [
    "h_njet",
    "HT",
    "pt_1",
    "pt_2",
    "m14_m",
    "m14_vs_m24",
    "m14_vs_m13",
    "ratio_m14_vs_m24",
    "ratio_m14_vs_m13",
    ]

for p in toplot:
    plotter(p, sig, normalize=True)
