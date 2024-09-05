import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting import plot1D, PlotObject, createPlotObject, loadStyles,plot2D
import analyzer.datasets as ds
import warnings
from analyzer.core import AnalysisResult
from rich.progress import track, Progress
import matplotlib.pyplot as plt
from pathlib import Path
from rich import print


# results_file = "results/histograms/selectionstudy_2018mc.pkl"
results_file = "results/histograms/crdata2018.pkl"

profile_repo = ds.ProfileRepo()
profile_repo.loadFromDirectory("profiles")
sample_manager = ds.SampleManager()
sample_manager.loadSamplesFromDirectory("datasets", profile_repo)

results = AnalysisResult.fromFile(results_file)
histograms = results.getMergedHistograms(sample_manager)


print(histograms)


axes_map = dict(
    [
        ("n_tightb", ("passes_1tightbjet",)),
        (
            "n_medb",
            (
                "passes_2bjet",
                "passes_1tightbjet",
                "passes_b_dr",
            ),
        ),
        (
            "n_tightb",
            (
                "passes_2bjet",
                "passes_1tightbjet",
                "passes_b_dr",
            ),
        ),
        (
            "med_b_dr",
            (
                "passes_b_dr",
                "passes_1tightbjet",
                "passes_2bjet",
            ),
        ),
        ("pt0", ("passes_highptjet",)),
        ("nlep", ("passes_0Lep",)),
        (
            "njets",
            (
                "passes_njets",
                "passes_2bjet",
                "passes_b_dr",
            ),
        ),
    ]
)

# axes_map = dict(
#     [
#         ("med_b_dr", ("passes_b_dr","passes_0Lep", "passes_1tightbjet")),
#     ]
# )

sel_names = [
    "passes_1tightbjet",
    "passes_2bjet",
    "passes_3bjet",
    "passes_0bjet",
    "passes_b_dr",
    "passes_hlt",
    "passes_njets",
    "passes_highptjet",
    "passes_0Lep",
]

sel_names = [
    "passes_hlt_PFHT1050",
    "passes_hlt_AK8PFJet400_TrimMass30",
    "passes_ht1200",
    "passes_jets",
    "passes_highptjet",
    "passes_0looseb",
    "passes_0Lep",
]

axes_map = dict(
    [
        ("pt0", ("passes_highptjet",)),
        ("nlep", ("passes_0Lep",)),
        ("njets", ("passes_jets",)),
        ("n_looseb", ("passes_0looseb",)),
        ("HT", ("passes_ht1200",)),
        ("phi_vs_eta", ("passes_ht1200",)),
        ("phi_vs_eta", ("passes_ht1200", "passes_hlt_AK8PFJet400_TrimMass30")),
        ("phi_vs_eta", ("passes_ht1200", "passes_hlt_PFHT1050")),
        #("phi_vs_eta", ("passes_ht1200", "passes_hlt_PFHT1050",)),
    ]
)

loadStyles()

sigs = [
    "signal_312_2000_1900",
    "signal_312_1200_1000",
    "signal_312_1500_1400",
    "signal_312_1200_400",
    "signal_312_2000_900",
    "signal_312_1500_1200",
]


def makeNMinusOnePlot(hname, axis_name, hists, output):
    def isSignal(s):
        return "signal" in s

    c = {i: sum if i in axis_name else True for i in sel_names}
    c.update(
        {
            # "passes_njets" : True,
            # "passes_highptjet" : True,
            # "passes_0Lep" : True,
            # "passes_hlt": True,
            # "passes_3bjet": sum,
        }
    )
    b = [
        createPlotObject(s, h[c], sample_manager).normalize()
        for s, h in hists.items()
        if not isSignal(s)
    ]

    s = [
        createPlotObject(s, h[c], sample_manager).normalize()
        for s, h in hists.items()
        if s in sigs
    ]
    output.mkdir(exist_ok=True, parents=True)
    if len((s + b)[0].axes) == 1:
        fig = plot1D(s, b, 59.8, 312, 2018, scale="linear")
        aname = "_".join(axis_name)
        fig.savefig(output / f"{hname}_WITHOUT_{aname}.pdf")
        plt.close(fig)
    elif len((s + b)[0].axes) == 2:
        for p in s+b:
            fig = plot2D(p, "59.8", "312", "2018", scale="linear")
            aname = "_".join(axis_name)
            fig.savefig(output / f"{hname}_{p.title}_WITHOUT_{aname}.pdf")
            plt.close(fig)


def makePlainPlot(hname, hists, output):
    def isSignal(s):
        return "signal" in s

    c = {i: sum for i in sel_names}
    c.update(
        {
            # "passes_njets" : True,
            # "passes_highptjet" : True,
            # "passes_0Lep" : True,
            # "passes_hlt": True,
        }
    )
    b = [
        createPlotObject(s, h[c], sample_manager).normalize()
        for s, h in hists.items()
        if not isSignal(s)
    ]
    s = [
        createPlotObject(s, h[c], sample_manager).normalize()
        for s, h in hists.items()
        if s in sigs
    ]
    output.mkdir(exist_ok=True, parents=True)
    if len((s + b)[0].axes) == 1:
        fig = plot1D(s, b, 59.8, 312, 2018, scale="linear")
        fig.savefig(output / f"{hname}.pdf")
        plt.close(fig)
    elif len((s + b)[0].axes) == 2:
        for p in s+b:
            fig = plot2D(p, "59.8", "312", "2018", scale="linear")
            fig.savefig(output / f"{hname}_{p.title}.pdf")
            plt.close(fig)


for hname, axis in track(axes_map.items()):
    makeNMinusOnePlot(
        hname, axes_map[hname], histograms[hname], Path("figures/selection")
    )
    makePlainPlot(hname, histograms[hname], Path("figures/selection"))
