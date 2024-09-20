import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Union,
)


import awkward as ak
import hist
import hist.dask as dah

from .configuration import HistogramSpec
from analyzer.utils.structure_tools import accumulate
from pydantic import BaseModel, ConfigDict

Hist = Union[hist.Hist, dah.Hist]

logger = logging.getLogger(__name__)

class HistogramCollection(BaseModel):
    """A histogram needs to keep track of both its nominal value and any variations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: HistogramSpec
    histogram: Hist
    variations: dict[str, Hist]

    def __add__(self, other):
        if self.spec != other.spec:
            raise ValueError(f"Cannot add two incomatible histograms")
        return HistogramCollection(
            spec=self.spec,
            histogram=self.histogram + other.histogram,
            variations=accumulate([self.variations, other.variations]),
        )


def transformToFill(fill_data, per_event_value, mask=None):
    if mask is None:
        if fill_data.ndim == 2:
            if ak.sum(ak.is_none(fill_data, axis=-1)) > 0:
                fill_data = ak.fill_none(fill_data, 0.0)
            return ak.flatten(fill_data * (per_event_value))
        else:
            return per_event_value

    if mask.ndim == 1 and not (fill_data is None) and fill_data.ndim == 2:
        if ak.sum(ak.is_none(fill_data, axis=-1)) > 0:
            fill_data = ak.fill_none(fill_data, 0.0)
        return ak.flatten(fill_data * (per_event_value[mask]))
    elif mask.ndim == 2:
        return (ak.flatten((ak.ones_like(mask) * per_event_value)[mask]),)
    else:
        return per_event_value[mask]


def maybeFlatten(data):
    if data.ndim == 2:
        return ak.flatten(data)
    else:
        return data


def makeHistogram(spec, category_info, fill_data, weight, mask=None):
    if len(spec.axes) == 1:
        fill_data = [fill_data]

    assert len(fill_data) == len(spec.axes)
    represenative = fill_data[0]
    all_axes = [x.axis for x in category_info] + spec.axes
    cat_values = [transformToFill(represenative, x.values, mask) for x in category_info]
    all_values = cat_values + [maybeFlatten(x) for x in fill_data]

    real_weight = transformToFill(represenative, weight, mask)
    histogram = dah.Hist(*all_axes, storage=spec.storage)
    histogram.fill(*all_values, weight=real_weight)
    return histogram


def generateHistogramCollection(
    spec,
    fill_data,
    categories,
    weight_repo,
    mask=None,
):

    variations = spec.variations
    include = spec.weights or []
    central_weight = weight_repo.weight(modifier=None)
    base_histogram = makeHistogram(spec, categories, fill_data, central_weight, mask)
    vh = {}
    logger.debug(f"Creating histogram collection \"{spec.name}\":\n Weights: {include}\nVariations:{variations}")
    for variation in variations:
        w = weight_repo.weight(modifier=variation)
        h = makeHistogram(spec, categories, fill_data, w, mask)
        vh[variation] = h
    return HistogramCollection(spec=spec, histogram=base_histogram, variations=vh)


