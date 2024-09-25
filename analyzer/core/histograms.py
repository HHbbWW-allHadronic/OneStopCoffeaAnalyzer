import copy
import logging
from typing import Any, Union, Optional

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
    include_unweighted: bool = True


class HistogramCollection(BaseModel):
    """A histogram needs to keep track of both its nominal value and any variations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: HistogramSpec
    histogram: Hist
    variations: dict[str, Hist]
    unweighted: Optional[Hist] = None

    def __add__(self, other):
        if self.spec != other.spec:
            raise ValueError(f"Cannot add two incomatible histograms")
        if self.unweighted is not None:
            new_un = self.unweighted + other.unweighted
        else:
            new_un = None
        return HistogramCollection(
            spec=self.spec,
            histogram=self.histogram + other.histogram,
            variations=accumulate([self.variations, other.variations]),
            unweighted=new_un,
        )

    def scaled(self, scale):
        if self.spec.no_scale:
            return self
        return HistogramCollection(
            spec=self.spec,
            histogram=self.histogram * scale,
            variations={x: scale * y for x, y in self.variations.items()},
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
    histogram = dah.Hist(*all_axes, storage=spec.storage)
    if weight is not None:
        real_weight = transformToFill(represenative, weight, mask)
        histogram.fill(*all_values, weight=real_weight)
    else:
        histogram.fill(*all_values)
    return histogram


def generateHistogramCollection(
    spec,
    fill_data,
    categories,
    weight_repo=None,
    mask=None,
):

    variations = spec.variations
    include = spec.weights or []
    if weight_repo:
        central_weight = weight_repo.weight(modifier=None)
        logger.debug(
            f'Creating histogram collection "{spec.name}":\nWeights: {include}\nVariations:{variations}'
        )
    else:
        logger.debug(f'Creating UNWEIGHTED histogram collection "{spec.name}"')
        central_weight = None

    base_histogram = makeHistogram(spec, categories, fill_data, central_weight, mask)
    print(dak.necessary_columns(base_histogram))
    if not weight_repo:
        return HistogramCollection(spec=spec, histogram=base_histogram, variations={})
    if spec.include_unweighted:
        un_histogram = makeHistogram(spec, categories, fill_data, None, mask)

    vh = {}
    for variation in variations:
        w = weight_repo.weight(modifier=variation)
        h = makeHistogram(spec, categories, fill_data, w, mask)
        vh[variation] = h
    return HistogramCollection(
        spec=spec, histogram=base_histogram, variations=vh, unweighted=un_histogram
    )
