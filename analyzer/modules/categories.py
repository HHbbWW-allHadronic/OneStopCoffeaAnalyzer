import awkward as ak
import hist

from analyzer.core import analyzerModule
from analyzer.modules.axes import *


@analyzerModule("dataset_category", categories="category", depends_on=["objects"])
def datasetCategory(events, analyzer):
    analyzer.histogram_builder.addCategory(dataset_axis, analyzer.setname)
    return events, analyzer


@analyzerModule("njets_category", depends_on=["objects"], categories="category")
def njetCategory(events, analyzer):
    # a = hist.axis.IntCategory([4, 5, 6], name="number_jets", label="NJets")
    return events, analyzer
