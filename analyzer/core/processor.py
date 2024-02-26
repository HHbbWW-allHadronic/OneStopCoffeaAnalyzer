from .results import DatasetDaskRunResult
from analyzer.histogram_builder import HistogramBuilder
from coffea.analysis_tools import PackedSelection, Weights

import logging


logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(
        self,
        dask_result: DatasetDaskRunResult,
        setname: str,
        delayed=True,
    ):
        self.setname = setname
        self.dask_result = dask_result
        self.delayed = delayed

        self.__selection = PackedSelection()
        self.__weights = Weights(None)

        self.histogram_builder = HistogramBuilder(self.weights)

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
