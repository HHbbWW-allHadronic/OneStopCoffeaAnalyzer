import logging
from typing import Any

from analyzer.histogram_builder import HistogramBuilder
from coffea.analysis_tools import PackedSelection, Weights

from .results import DatasetDaskRunResult

logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(
        self,
        dask_result: DatasetDaskRunResult,
        dataset_name: str,
        setname: str,
        profile: Any,
        delayed=True,
        skim_save_path=None,
    ):
        self.dataset_name = dataset_name
        self.setname = setname
        self.dask_result = dask_result
        self.delayed = delayed
        self.profile = profile

        self.__selection = PackedSelection()
        self.__weights = Weights(None)

        self.histogram_builder = HistogramBuilder(self.weights)

        self.skim_save_path = skim_save_path
        self.skim_save_cols = ["HLT", "Jet", "Electron", "Muon", "FatJet"]
        self.side_effect_computes= None

    @property
    def selection(self):
        return self.__selection

    @property
    def histograms(self):
        return self.dask_result.histograms

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, val):
        self.__weights = val

    def applySelection(self, events):
        if self.selection.names:
            events = events[self.selection.all(*self.selection.names)]
        return events

    def maybeCreateAndFill(
        self,
        key,
        axis,
        data,
        mask=None,
        name=None,
        description=None,
        auto_expand=True,
    ):
        name = name or key
        if key not in self.histograms:
            self.histograms[key] = self.histogram_builder.createHistogram(
                axis, name, description, delayed=self.delayed
            )
        self.histogram_builder.fillHistogram(
            self.histograms[key], data, mask, event_weights=self.weights
        )

    def H(self, *args, **kwargs):
        return self.maybeCreateAndFill(*args, **kwargs)
