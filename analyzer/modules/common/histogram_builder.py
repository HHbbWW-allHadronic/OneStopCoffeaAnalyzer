from attrs import define

from analyzer.core.columns import Column, TrackedColumns, EventBackend
import numpy as np
import awkward as ak
from analyzer.core.results import Histogram
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    ModuleAddition,
    PureResultModule,
)
from .axis import Axis, RegularAxis
import hist.dask as dah
import hist
import logging

logger = logging.getLogger("analyzer.modules")


@define
class HistogramBuilder(PureResultModule):
    product_name: str
    columns: list[Column]
    axes: list[Axis]
    storage: str = "weight"
    mask_col: Column | None = None

    @staticmethod
    def transformToFill(fill_data, per_event_value, mask=None):
        """
        Perform transformations to bring fill data to correct shape
        """
        if mask is None:
            if fill_data.ndim == 2:
                fill_data = ak.ones_like(fill_data, dtype=np.int32)
                fill_data = ak.fill_none(fill_data, 0)
                r = ak.flatten(fill_data * per_event_value)
                return r
            else:
                return per_event_value

        if mask.ndim == 1 and fill_data is not None and fill_data.ndim == 2:
            fill_data = ak.ones_like(fill_data, dtype=np.int32)
            fill_data = ak.fill_none(fill_data, 0)
            return ak.flatten(fill_data * per_event_value[mask])

        elif mask.ndim == 2:
            return ak.flatten((ak.ones_like(mask) * per_event_value)[mask])

        else:
            return per_event_value[mask]

    @staticmethod
    def maybeFlatten(data):
        if data.ndim == 2:
            return ak.flatten(data)
        else:
            return data

    @staticmethod
    def fillHistogram(
        histogram,
        cat_values,
        fill_data,
        weight=None,
        variation="central",
        mask=None,
    ):
        all_values = (
            [variation]
            + cat_values
            + [HistogramBuilder.maybeFlatten(x) for x in fill_data]
        )
        if weight is not None:
            histogram.fill(*all_values, weight=weight)
        else:
            histogram.fill(*all_values)
        return histogram

    @staticmethod
    def create(backend, categories, axes, storage):
        variations_axis = hist.axis.StrCategory([], name="variation", growth=True)
        all_axes = (
            [variations_axis]
            + [x.axis.toHist() for x in categories]
            + [x.toHist() for x in axes]
        )
        if backend == EventBackend.coffea_dask:
            histogram = dah.Hist(*all_axes, storage=storage)
        else:
            histogram = hist.Hist(*all_axes, storage=storage)
        return histogram

    def run(self, column_sets, params):
        if isinstance(column_sets, TrackedColumns):
            column_sets = [["central", column_sets]]

        backend = column_sets[0][1].backend
        pipeline_data = column_sets[0][1].pipeline_data
        categories = pipeline_data.get("categories", {})
        histogram = HistogramBuilder.create(
            backend, categories, self.axes, self.storage
        )
        logger.debug(
            f"Creating histogram {self.product_name} with the following variations:\n{[x[0] for x in column_sets]}"
        )
        for cset in column_sets:
            name, columns = cset
            if self.mask_col is not None:
                mask = columns[self.mask_col]
            data_to_fill = [columns[x] for x in self.columns]
            if self.mask_col is not None:
                data_to_fill = [col[mask] for col in data_to_fill]
            representative = data_to_fill[0]
            mask = None

            if "Weights" in columns.fields:
                weights = columns["Weights"]
                total_weight = ak.prod([weights[x] for x in weights.fields], axis=0)
                total_weight = HistogramBuilder.transformToFill(
                    representative, total_weight, mask
                )
            else:
                total_weight = None

            cat_to_fill = [
                HistogramBuilder.transformToFill(
                    representative, columns[x.column], mask
                )
                for x in categories
            ]
            HistogramBuilder.fillHistogram(
                histogram,
                cat_to_fill,
                data_to_fill,
                weight=total_weight,
                variation=name,
            )

        return [Histogram(name=self.product_name, histogram=histogram, axes=self.axes)]

    def inputs(self, metadata):
        if self.mask_col is not None:
            return [
                *self.columns,
                self.mask_col,
                Column("Categories"),
                Column("Weights"),
            ]
        else:
            return [*self.columns, Column("Categories"), Column("Weights")]

    def outputs(self, metadata):
        return []


def makeHistogram(
    product_name: str,
    columns,
    axes: Axis | list[Axis],
    data,
    description=None,
    mask=None,
):
    """
    Create a histogram from column data and register it in the pipeline.

    This helper function wraps input data into temporary columns, associates
    them with axes definitions, and builds a histogram that can be added to
    the module outputs.

    Parameters
    ----------
    product_name : str
        Name of the histogram/product to create.
    columns : list[Column]
        Collection of event data columns where temporary columns will be stored.
    axes : Axis or list of Axis
        Axis (or axes) definition(s) for the histogram. Must match the dimensionality of `data`.
    data : array-like or list of array-like
        Data array(s) to histogram. Can be a single array or a list/tuple for multiple dimensions.
    description : str, optional
        Optional description for the histogram.
    mask : array-like, optional
        Boolean array indicating which entries should be included in the histogram.

    Returns
    -------
    ModuleAddition
        Object encapsulating the histogram builder, ready to be added to
        an analyzer module's outputs.

    Notes
    -----
    - Temporary columns are created for each data array to integrate with the pipeline.
    - If `mask` is provided, it is stored in a separate column and used by the histogram builder.
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
        axes = [axes]

    names = []
    for i, d in enumerate(data):
        name = Column(f"INTERNAL_USE.auto-col-{product_name}-{i}")
        names.append(name)
        columns[name] = d

    if mask is not None:
        mask_col_name = Column(f"INTERNAL_USE.mask-{product_name}")
        columns[mask_col_name] = mask
    else:
        mask_col_name = None

    b = HistogramBuilder(product_name, names, axes, mask_col=mask_col_name)
    return ModuleAddition(b)


@define
class SimpleHistogram(AnalyzerModule):
    hist_name: str
    input_cols: list[Column]
    axes: list[RegularAxis]

    def outputs(self, metadata):
        return []

    def inputs(self, metadata):
        return self.input_cols

    def run(self, columns, params):
        data = [columns[x] for x in self.input_cols]
        h = makeHistogram(
            self.hist_name,
            columns,
            self.axes,
            data,
        )
        return columns, [h]
