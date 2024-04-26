import sys
sys.path.append("/srv")

from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting.plots_2d import AnnotRectangle, drawRectangles
from analyzer.datasets import SampleManager
from analyzer.core import AnalysisResult

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

file_name="signal24-04-26.pkl"

sample_manager = SampleManager()
sample_manager.loadSamplesFromDirectory("datasets")

res = AnalysisResult.fromFile(file_name)
histos = res.getMergedHistograms(sample_manager)


def histMean(histogram):
    centers = histogram.axes[0].centers
    vals=histogram.values()
    mean = np.average(centers, weights=vals)
    return mean

def histVariance(histogram):
    centers = histogram.axes[0].centers
    mean = np.average(centers, weights=histogram.values())
    return np.average((centers - mean)**2, weights=histogram.values())

def makeRecoPlot(hname, hist, savedir ,prefix=""):
    savedir=Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    signals = [x for x in hist.axes[0] if "signal" in x]

    rects = []
    for s in signals:
        h = hist[s,...]
        _,_,mt,mx=s.split("_")
        mt,mx=int(mt),int(mx)
        mean = histMean(h)
        stddev = np.sqrt(histVariance(h))
        v = (abs(mean - mx) + stddev)/mx
        r = AnnotRectangle(x=mt,y=mx, w=50,h=50, text=f"{round(v,2)}\n({round(mean)})", value=v)
        rects.append(r)
    fig, ax = plt.subplots()
    drawRectangles(ax, rects)
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi^{\pm}}}$")
    ax.cax.set_ylabel(r"$\frac{|m_{reco} - m_{exp}| + \sigma_{reco}}{m_{exp}}$")
    fig.tight_layout()
    fig.savefig(savedir /( (prefix or "") + hname + ".pdf"))

def makeMatchPlot(hname, target_range, savedir ,prefix=""):
    savedir=Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    hist = histos[hname]
    signals = [x for x in hist.axes[0] if "signal" in x]

    rects = []
    for s in signals:
        _,_,mt,mx=s.split("_")
        mt,mx=int(mt),int(mx)
        h = hist[s,...]
        total = h.sum().value
        good=h[target_range].value
        ratio = good/total
        r = AnnotRectangle(x=mt,y=mx, w=50,h=50, text=f"{round(ratio,2)}", value=ratio)
        rects.append(r)
    fig, ax = plt.subplots()
    drawRectangles(ax, rects)
    ax.set_xlabel(r"$m_{\tilde{t}}$")
    ax.set_ylabel(r"$m_{\tilde{\chi^{\pm}}}$")
    ax.cax.set_ylabel(r"Percent All Correct")
    fig.savefig(savedir /( (prefix or "") + hname + ".pdf"))

def makeReco(hname):
    makeRecoPlot(hname, histos[hname], "figures", "recoplot_")

makeReco("m24_m")
makeReco("m13_m")
makeReco("m3_top_3_no_lead_b")
makeReco("m3_top_3_no_lead_b")
#makeMatchPlot("m13_matching_all_three", 3, "figures", "gridplot_")
#makeMatchPlot("m24_matching_all_three", 3, "figures", "gridplot_")
#makeMatchPlot("m3_top_2_plus_lead_b_matching_all_three", 3, "figures", "gridplot_")
#makeMatchPlot("m3_top_3_no_lead_b_matching_all_three", 3, "figures", "gridplot_")
    
        
