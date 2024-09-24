import copy
import logging
from typing import Any, Optional, Union

import awkward as ak
import hist
import hist.dask as dah
from analyzer.utils.structure_tools import accumulate
from pydantic import BaseModel, ConfigDict

Hist = Union[hist.Hist, dah.Hist]

logger = logging.getLogger(__name__)


class HistogramSpec(BaseModel):
    name: str
    axes: list[Any]
    storage: str = "weight"
    description: str
    weights: list[str]
    variations: list[str]
    no_scale: bool = False


class HistogramCollection(BaseModel):
    """A histogram needs to keep track of both its nominal value and any variations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: HistogramSpec
    histogram: Hist

    def __add__(self, other):
        if self.spec != other.spec:
            raise ValueError(f"Cannot add two incomatible histograms")
        return HistogramCollection(
            spec=self.spec,
            histogram=self.histogram + other.histogram,
        )

    def scaled(self, scale):
        if self.spec.no_scale:
            return self
        return HistogramCollection(
            spec=self.spec,
            histogram=self.histogram * scale,
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


def fillHistogram(
    histogram, cat_values, fill_data, weight, variation_val=None, mask=None
):
    if variation_val:
        all_values = [variation_val] + cat_values + [maybeFlatten(x) for x in fill_data]
    else:
        all_values = cat_values + [maybeFlatten(x) for x in fill_data]
    if weight is not None:
        histogram.fill(*all_values, weight=weight)
    else:
        histogram.fill(*all_values)
    return histogram


def __unweightedCollection(
    spec,
    fill_data,
    categories,
    mask=None,
):
    represenative = fill_data[0]
    all_axes = [x.axis for x in categories] + spec.axes
    cat_values = [transformToFill(represenative, x.values, mask) for x in categories]
    histogram = dah.Hist(*all_axes, storage=spec.storage)
    fillHistogram(histogram, cat_values, fill_data, None, mask=mask)
    return HistogramCollection(spec=spec, histogram=histogram)


def __weightedCollection(
    spec,
    fill_data,
    categories,
    weight_repo,
    mask=None,
):
    variations = spec.variations
    has_variations = variations is not None
    represenative = fill_data[0]

    assert has_variations == (weight_repo is not None)
    include_weights = spec.weights or []
    central_weight = weight_repo.weight(modifier=None)
    logger.debug(
        f'Creating histogram collection "{spec.name}":\nWeights: {include_weights}\nVariations:{variations}'
    )

    variations_axis = hist.axes.StrCategory(["central", *variations], name="variation")
    all_axes = [variations_axis] + [x.axis for x in categories] + spec.axes
    cat_values = [transformToFill(represenative, x.values, mask) for x in categories]
    histogram = dah.Hist(*all_axes, storage=spec.storage)
    central_weight = transformToFill(represenative, central_weight, mask)
    fillHistogram(histogram, cat_values, fill_data, central_weight, variation_val="central", mask=mask)
    for variation in variations:
        w = weight_repo.weight(modifier=variation)
        real_weight = transformToFill(represenative, w, mask)
        fillHistogram(
            histogram,
            cat_values,
            fill_data,
            real_weight,
            variation_val=variation,
            mask=mask,
        )
    return HistogramCollection(spec=spec, histogram=histogram)


def generateHistogramCollection(
    spec,
    fill_data,
    categories,
    weight_repo=None,
    mask=None,
):

    if not isinstance(fill_data, (list, tuple)):
        fill_data = [fill_data]
    if weight_repo is None:
        return __unweightedCollection(spec, fill_data, categories, mask=mask)
    else:
        return __weightedCollection(spec, fill_data, categories, weight_repo, mask=mask)
