# import awkward as ak
import logging

import awkward as ak
import dask_awkward as dak
import numpy as np
import hist
import hist.dask as hda

logger = logging.getLogger(__name__)


class HistogramBuilder:
    def __init__(self, category_axes=None, category_values=None):
        self.cat_axes = list(category_axes or [])
        self.cat_values = list(category_values or [])

    def addCategory(self, axis, value):
        logger.info(f"Adding axis with name {axis.name} to histogram builder.")
        self.cat_axes.append(axis)
        self.cat_values.append(value)

    def fillHistogram(self, histogram, data, mask=None, event_weights=None):
        logger.debug(
            f"Filling histogram {histogram.name} with data {data} and mask {mask}"
        )
        weights = event_weights
        if not isinstance(data, list):
            data = [data]
        if not data:
            raise Exception("No data")
        h = histogram

        if (weights is not None) and (mask is not None):
            weights = weights[mask]

        base_category_vals = self.cat_values

        if mask is not None:
            base_category_vals = [
                x[mask] if isinstance(x, (ak.Array, dak.Array, np.ndarray)) else x
                for x in base_category_vals
            ]

        shaped_cat_vals = base_category_vals
        shaped_data_vals = data

        dtype_cache = {}

        def expandWithType(template_data, other):
            t = str(ak.type(other))
            if t in dtype_cache:
                return dtype_cache[t]
            else:
                ret = ak.ones_like(template_data, dtype=t)
                dtype_cache[t] = ret
                return ret

        template_data = data[0]
        if template_data.ndim == 2:
            ol = ak.ones_like(template_data)
            if weights is not None:
                weights = ak.flatten(ol * weights)
            shaped_cat_vals = [
                (
                    ak.flatten(expandWithType(template_data, x) * x)
                    if isinstance(x, (ak.Array, dak.Array, np.ndarray))
                    else x
                )
                for x in base_category_vals
            ]
            shaped_data_vals = [
                ak.flatten(x) if isinstance(x, (ak.Array, dak.Array, np.ndarray)) else x
                for x in shaped_data_vals
            ]

        d = shaped_cat_vals + shaped_data_vals


        if weights is not None:
            ret = h.fill(*d, weight=weights)
        else:
            ret = h.fill(*d)
        return ret

    def createHistogram(
        self,
        axis,
        name=None,
        description=None,
        delayed=True,
    ):
        logger.debug(f"Creating histogram with axes {axis} and name {name}")

        if isinstance(axis, list):
            all_axes = self.cat_axes + list(axis)
        else:
            all_axes = self.cat_axes + [axis]

        if delayed:
            h = hda.Hist(*all_axes, storage="weight", name=name)
        else:
            h = hist.Hist(*all_axes, storage="weight", name=name)
        setattr(h, "description", description)
        setattr(h, "name", name)
        return h

    def __call__(self, *args, **kwargs):
        return self.makeHistogram(*args, **kwargs)
