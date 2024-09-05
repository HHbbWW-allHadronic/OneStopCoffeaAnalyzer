import logging
from typing import Any
import hist.dask as dah
import itertools as it
import awkward as ak
import dask_awkward as dak


from analyzer.histogram_builder import HistogramBuilder
from coffea.analysis_tools import PackedSelection, Weights
import numpy as np

from .results import DatasetDaskRunResult

logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(
        self,
        dask_result: DatasetDaskRunResult,
        dataset_name: str,
        setname: str,
        last_ancestor: str,
        profile: Any,
        delayed=True,
        skim_save_path=None,
    ):
        self.dataset_name = dataset_name
        self.last_ancestor = last_ancestor
        self.setname = setname
        self.dask_result = dask_result
        self.delayed = delayed
        self.profile = profile

        self.processing_info = {}

        self.__selection = PackedSelection()
        self.__weights = Weights(None)

        self.histogram_builder = HistogramBuilder()

        self.skim_save_path = skim_save_path
        self.skim_save_cols = [
            "HLT",
            "Jet",
            "Electron",
            "Muon",
            "FatJet",
            "run",
            "luminosityBlock",
            "event",
        ]
        self.side_effect_computes = None

        # {"WeightName" : {"central" : "Arary", "systs": {"SystName" : (Up, Down)}}}
        self.presel_weights = {}
        self.postsel_weights = {}
        self.is_pre_selection = True
        self.selection_mask = None
        self.required_columns = set()
        self.critical_selections = []

    @property
    def selection(self):
        return self.__selection

    @property
    def histograms(self):
        return self.dask_result.histograms

    @property
    def nshistograms(self):
        return self.dask_result.non_scaled_histograms

    @property
    def nshistogramslabels(self):
        return self.dask_result.non_scaled_histograms_labels

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, val):
        self.__weights = val

    def applySelection(self, events):
        if self.processing_info.get("apply_noncritical_selections", True):
            names = self.selection.names
        else:
            logger.info(
                f'Since "apply_noncritical_selections" is False, selections will be limited. Critcal selections are: {self.critical_selections}.'
            )
            if self.selection.names is not None:
                names = [
                    x for x in self.selection.names if x in self.critical_selections
                ]
            else:
                names = None
        logger.info(f"Applying the following selections:\n{names}")
        if names:
            sm = None
            hlts = [x for x in names if x.startswith("hlt")]
            rest = [x for x in names if not x.startswith("hlt")]
            hlt_mask = self.selection.any(*hlts) if hlts else None
            rest_mask = self.selection.all(*rest) if rest else None
            if hlt_mask and rest_mask:
                self.selection_mask = hlt_mask & rest_mask
            elif hlt_mask:
                self.selection_mask = hlt_mask
            else:
                self.selection_mask = rest_mask
            events = events[self.selection_mask]
            logger.info("Applied selections")
        else:
            self.selection_mask = ~ak.is_none(events.event)
        self.is_pre_selection = False
        return events

    def addWeight(self, name, central, systs=None):
        if self.is_pre_selection:
            logger.debug(f'Adding pre-selection weight "{name}\:')
            self.presel_weights[name] = {"central": central, "systs": systs or {}}
        else:
            logger.debug(f'Adding post-selection weight "{name}"')
            self.postsel_weights[name] = {"central": central, "systs": systs or {}}

    def __addMultiWeight(self, data, mask=None):
        for wname, vals in data.items():
            logger.debug(f"Adding event weight {wname} to dataset {self.dataset_name}")
            if "systs" in vals:
                systs = [(x, *y) for x, y in vals["systs"].items()]
                name, up, down = list(map(list, zip(*systs)))
                logger.debug(f"Weight {wname} has variations {', '.join(name)}")
                self.__weights.add_multivariation(
                    wname,
                    vals["central"][mask] if mask is not None else vals["central"],
                    name,
                    [x[mask] for x in up] if mask is not None else up,
                    [x[mask] for x in down] if mask is not None else down,
                )
            else:
                systs = []
                self.__weights.add_multivariation(
                    wname,
                    vals["central"][mask] if mask else vals["central"],
                    [],
                    [],
                    [],
                )

    def finalizeWeights(self):
        self.__addMultiWeight(self.presel_weights, mask=self.selection_mask)
        logger.info(f"Finalized pre-selection weights with selection mask")
        self.__addMultiWeight(self.postsel_weights)
        logger.info(f"Finalized post selection weights")

        s = "\n".join(
            [
                f"{i+1}. {x}"
                for i, x in enumerate((*self.presel_weights, *self.postsel_weights))
            ]
        )
        logger.info(
            f"The following weights will be used for sample {self.dataset_name}:\n{s}"
        )
        s = "\n".join([f"{i+1}. {x}" for i, x in enumerate(self.__weights.variations)])

        logger.info(
            f"The following variations will be used for sample {self.dataset_name}:\n{s}"
        )

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
        logger.debug(f'Creating new histogram "{name}"')

        def makeAndFillWithWeight(key, name, weight):
            if key not in self.histograms:
                logger.debug(f'Histogram "{name}" is not yet present, creating now.')
                self.histograms[key] = self.histogram_builder.createHistogram(
                    axis, name, description, delayed=self.delayed
                )
            logger.debug(f'Filling histogram "{name}".')
            self.histogram_builder.fillHistogram(
                self.histograms[key], data, mask, event_weights=weight
            )

        variations = self.__weights.variations
        base_w = self.weights.weight()
        if base_w is None:
            makeAndFillWithWeight(key, name, None)
        else:
            makeAndFillWithWeight(
                f"unweighted_{key}", name, ak.ones_like(self.weights.weight())
            )
            makeAndFillWithWeight(key, name, self.weights.weight())
            for v in variations:
                makeAndFillWithWeight(
                    key + "_" + v, name + "_" + v, self.weights.weight(v)
                )

    def H(self, *args, **kwargs):
        return self.maybeCreateAndFill(*args, **kwargs)

    def add_non_scaled_hist(self, key: str, hist: dah.Hist, labels: list):
        logger.debug(f"Adding non scaled histogram {key}")
        if key not in self.nshistograms:
            self.nshistogramslabels[key] = labels
            self.nshistograms[key] = hist
