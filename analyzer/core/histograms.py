import logging
import operator as op
from typing import Any, Union, Optional

import awkward as ak
import hist
import hist.dask as dah
from pydantic import BaseModel, ConfigDict
import dask_awkward as dak

Hist = Union[hist.Hist, dah.Hist]

logger = logging.getLogger(__name__)


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
    histogram, cat_values, fill_data, weight=None, variation_val="central", mask=None
):
    all_values = [variation_val] + cat_values + [maybeFlatten(x) for x in fill_data]
    if weight is not None:
        histogram.fill(*all_values, weight=weight)
    else:
        histogram.fill(*all_values)
    return histogram


class HistogramSpec(BaseModel):
    # model_config = ConfigDict(frozen=True)
    name: str
    axes: list[Any]
    storage: str = "weight"
    description: str
    variations: Optional[list[str]] = None
    store_unweighted: bool = True
    no_scale: bool = False

    def model_post_init(self, __context):
        if self.variations:
            self.variations = sorted(self.variations)


class HistogramCollection(BaseModel):
    """A histogram needs to keep track of both its nominal value and any variations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: HistogramSpec
    histogram: Hist

    def __add__(self, other):
        if self.spec != other.spec:
            logger.error(
                "Cannot add two incompatible histograms specs. Hist1:\n"
                f"{self.spec}\nHist2:\n{other.spec}"
            )
            raise ValueError(f"Cannot add two incomatible histograms")
        return HistogramCollection(
            spec=self.spec,
            histogram=self.histogram + other.histogram,
        )

    def get(self, variation=None):
        has_variations = "variation" in map(op.attrgetter("name"), self.histogram.axes)
        if variation:
            return self.histogram[variation, ...]
        else:
            return self.histogram["central", ...]

    def scaled(self, scale):
        if self.spec.no_scale:
            return self
        return HistogramCollection(
            spec=self.spec,
            histogram=self.histogram * scale,
        )

    def fill(
        self,
        fill_data,
        categories,
        weight_repo,
        active_shape_systematic=None,
        mask=None,
    ):
        if not isinstance(fill_data, (list, tuple)):
            fill_data = [fill_data]

        represenative = fill_data[0]
        if active_shape_systematic is not None:
            weight_variations = []
            central_name = active_shape_systematic
        else:
            central_name = "central"
            if self.spec.variations is None:
                weight_variations = weight_repo.variations
            else:
                weight_variations = self.spec.variations

        central_weight = weight_repo.weight(modifier=None)

        if central_weight is not None:
            central_weight = transformToFill(represenative, central_weight, mask)

        cat_values = [
            transformToFill(represenative, x.values, mask) for x in categories
        ]
        fillHistogram(
            self.histogram,
            cat_values,
            fill_data,
            central_weight,
            variation_val=central_name,
            mask=mask,
        )
        for weight_variation in weight_variations:
            logger.debug(f'Filling histogram with variation "{weight_variation}"')
            w = weight_repo.weight(modifier=weight_variation)
            real_weight = transformToFill(represenative, w, mask)
            fillHistogram(
                self.histogram,
                cat_values,
                fill_data,
                real_weight,
                variation_val=weight_variation,
                mask=mask,
            )

    @staticmethod
    def create(spec, categories, delayed=True):
        variations_axis = hist.axis.StrCategory([], name="variation", growth=True)
        all_axes = [variations_axis] + [x.axis for x in categories] + spec.axes
        if delayed:
            histogram = dah.Hist(*all_axes, storage=spec.storage)
        else:
            histogram = hist.Hist(*all_axes, storage=spec.storage)
        return HistogramCollection(spec=spec, histogram=histogram)


class Histogrammer:
    def __init__(
        self,
        storage,
        weighter,
        categories=None,
        active_shape_systematic=None,
        delayed=True,
    ):
        self.weighter = weighter
        self.categories = categories
        self.active_shape_systematic = active_shape_systematic
        self.storage = storage
        self.delayed = delayed

    def H(
        self,
        name,
        axes,
        values,
        variations=None,
        weights=None,
        description="",
        no_scale=False,
        mask=None,
        storage="weight",
    ):
        if name not in self.storage:
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
            spec = HistogramSpec(
                name=name,
                axes=axes,
                storage=storage,
                description=description,
                variations=variations,
                no_scale=no_scale,
            )
            ret = HistogramCollection.create(spec, self.categories, delayed=self.delayed)
            self.storage[name] = ret
        else:
            ret = self.storage[name]

        if self.active_shape_systematic is not None:
            shape_sys = "__".join(self.active_shape_systematic)
        else:
            shape_sys = None

        ret.fill(
            values,
            self.categories,
            self.weighter,
            active_shape_systematic=shape_sys,
            mask=mask,
        )
        return ret
