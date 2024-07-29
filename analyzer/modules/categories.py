import awkward as ak
import functools
import hist
import operator as op

from analyzer.core import analyzerModule
from analyzer.modules.axes import *


@analyzerModule("selection_categories", categories="category", depends_on=["objects"])
def selectionCategories(events, analyzer):
    good_jets = events.good_jets
    fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    med_b = events.med_bs
    tight_b = events.tight_bs
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
    filled_med = ak.pad_none(med_b, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    hlt_names = analyzer.profile.hlt
    passes_highptjet = ak.fill_none(filled_jets[:, 0].pt > 300, False)
    passes_njets = (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)
    passes_0Lep = (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)
    passes_2bjet = ak.num(med_b) >= 2
    passes_3bjet = ak.num(med_b) >= 3
    passes_1tightbjet = ak.num(tight_b) >= 1
    passes_b_dr = med_dr > 1

    if "HLT" in events.fields:
        hlt_names = analyzer.profile.hlt
        passes_hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
        analyzer.histogram_builder.addCategory(
            hist.axis.Boolean(name="passes_hlt"), passes_hlt
        )
    analyzer.histogram_builder.addCategory(
        hist.axis.Boolean(name="passes_highptjet"), passes_highptjet
    )
    analyzer.histogram_builder.addCategory(
        hist.axis.Boolean(name="passes_njets"), passes_njets
    )
    analyzer.histogram_builder.addCategory(
        hist.axis.Boolean(name="passes_0Lep"), passes_0Lep
    )
    analyzer.histogram_builder.addCategory(
        hist.axis.Boolean(name="passes_2bjet"), passes_2bjet
    )
    analyzer.histogram_builder.addCategory(
        hist.axis.Boolean(name="passes_3bjet"), passes_3bjet
    )
    analyzer.histogram_builder.addCategory(
        hist.axis.Boolean(name="passes_1tightbjet"), passes_1tightbjet
    )
    analyzer.histogram_builder.addCategory(
        hist.axis.Boolean(name="passes_b_dr"), passes_b_dr
    )
    return events, analyzer
