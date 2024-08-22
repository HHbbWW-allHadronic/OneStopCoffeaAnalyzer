import itertools as it

import awkward as ak

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi
from analyzer.core.processor import DatasetProcessor
import hist
from hist import Hist
import numpy as np
import dask_awkward as dak
import hist.dask as hda

from .axes import *
from .utils import numMatching


@analyzerModule("pre_sel_hists", categories="presel")
def makePreSelectionHistograms(events, hmaker):
    if "LHE" not in events.fields:
        return {}
    ret = {}
    w = events.EventWeight
    # ret[f"LHEHT"] = hmaker(
    #    ht_axis,
    #    events.LHE.HT,
    #    w,
    #    name="Event LHE HT Preselection",
    #    description="HT of the LHE level event before any selections are applied",
    # )
    return ret


@analyzerModule("event_level_hists", categories="main", depends_on=["objects"])
def createEventLevelHistograms(events, analyzer):
    ret = {}
    analyzer.H(
        f"HT",
        makeAxis(60, 0, 3000, "HT", unit="GeV"),
        events.HT,
        name="Event HT",
        description="Sum of $p_T$ of good AK4 jets.",
    )
    return events, analyzer
    if "LHE" not in events.fields:
        return events, analyzer
    analyzer.H(
        f"nQLHE",
        makeAxis(10, 0, 10, "Quark Count LHE"),
        events.LHE.Nuds + events.LHE.Nc + events.LHE.Nb,
        name="Quark Count LHE",
        description="Number of LHE level Quarks",
    )
    analyzer.H(
        f"nJLHE",
        makeAxis(10, 0, 10, "Jet Count LHE"),
        events.LHE.Njets,
        name="Jet Count LHE",
        description="Number of LHE level Jets",
    )
    analyzer.H(
        f"nGLHE",
        makeAxis(10, 0, 10, "Gluon Count LHE"),
        events.LHE.Nglu,
        name="Gluon Count LHE",
        description="Number of LHE level gluons",
    )
    return events, analyzer


@analyzerModule("tag_hists", depends_on=["objects"])
def createTagHistograms(events, hmaker):
    ret = {}
    for name, wp in it.product(("tops", "bs", "Ws"), ("loose", "med", "tight")):
        ret[f"{name}_{wp}"] = hmaker(
            tencountaxis,
            ak.num(events[f"{wp}_{name}"], axis=1),
            name=f"Number of {wp} {name}",
        )
    for name, wp in it.product(("deep_top", "deep_W"), range(1, 5)):
        ret[f"{name}_{wp}"] = hmaker(
            tencountaxis,
            ak.num(events[f"{name}_wp{wp}"], axis=1),
            name=f"Number of wp{wp} {name}",
        )

@analyzerModule("cutflow",categories="post_selection")
def createCutflowHistogram(events,analyzer):

    # def cutflowHist(nevents,masks,name,size):

    #     if not analyzer.delayed:
    #         h = hist.Hist(dataset_axis,hist.axis.Integer(0, size, name=name))
    #         h.fill(analyzer.setname,np.arange(size, dtype=int), weight=nevents)

    #     else:
    #         h = hda.Hist(dataset_axis,hist.axis.Integer(0, size, name=name))
    #         setattr(h, "name", name)
    #         for i, weight in enumerate(masks, 1):
    #             h.fill(
    #                 analyzer.setname,
    #                 dak.full_like(weight, i, dtype=int), weight=weight
    #             )
    #         h.fill(analyzer.setname,dak.zeros_like(weight, dtype=int))

    #     return h

    selection = analyzer.selection
    n1hist, n1labels = selection.nminusone(*selection.names).yieldhist()
    honecut, hcutflow, cutflow_labels = selection.cutflow(*selection.names).yieldhist()

    # size = len(cutflow.labels)
    # nminusonesize = len(nminusone.labels)
    # n1hist = cutflowHist(nminusone.nev,nminusone.masks,name='N-1',size=nminusonesize)
    # hcutflow = cutflowHist(cutflow.nevcutflow,cutflow.maskscutflow,name='cutflow',size=size)
    # honecut = cutflowHist(cutflow.nevonecut,cutflow.masksonecut,name='onecut',size=size)

    analyzer.add_non_scaled_hist(key='N-1',hist=n1hist,labels=n1labels)
    analyzer.add_non_scaled_hist(key='cutflow',hist=hcutflow,labels=cutflow_labels)
    analyzer.add_non_scaled_hist(key='onecut',hist=honecut,labels=cutflow_labels)

    return events,analyzer
