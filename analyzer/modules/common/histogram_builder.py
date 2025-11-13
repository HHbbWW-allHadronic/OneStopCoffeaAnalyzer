from attrs import define
import abc

from analyzer.core.columns import Column
from analyzer.core.analysis_modules import AnalyzerModule
from .axis import Axis


@define
class HistogramBuilder(AnalyzerModule):
    product_name: str
    columns: list[Column]
    axes: list[Axis]
    central_name: str = "central"
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

        if mask.ndim == 1 and not (fill_data is None) and fill_data.ndim == 2:
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
            + [x.axes.toHist() for x in axes]
        )
        if backend == EventBackend.coffea_dask:
            histogram = dah.Hist(*all_axes, storage=storage)
        else:
            histogram = hist.Hist(*all_axes, storage=storage)
        return histogram

    def run(self, column_sets, params):
        backend = column_sets[0][1].backend
        pipeline_data = column_sets[0][1].pipeline_data
        categories = pipeline_data.get("categories", {})
        histogram = HistogramBuilder.create(
            backend, categories, self.axes, self.storage
        )
        for cset in column_sets:
            params = cset[0]
            noncentral = {
                k: v
                for k, v in params.getAllByName("variation").items()
                if v != self.central_name
            }
            if len(noncentral) == 0:
                variation_name = "central"
            elif len(noncentral) == 1:
                variation_name = next(iter(noncentral.values()))
            else:
                raise RuntimeError(f"Multiple active systematics {noncentral}")
            columns = cset[1]
            data_to_fill = [columns[x] for x in self.columns]
            represenative = data_to_fill[0]
            mask = None
            if self.mask_col is not None:
                mask = columns[self.mask_col]

            if "Weights" in columns.fields:
                weights = columns["Weights"]
                total_weight = ak.prod([weights[x] for x in weights.fields], axis=1)
                total_weight = HistogramBuilder.transformToFill(
                    represenative, weight, mask
                )
            else:
                total_weight = None

            cat_to_fill = [
                HistogramBuilder.transformToFill(represenative, columns[x.column], mask)
                for x in categories
            ]
            HistogramBuilder.fillHistogram(
                histogram,
                cat_to_fill,
                data_to_fill,
                weight=total_weight,
                variation=variation_name,
            )

        return None, [Histogram(name=self.product_name, histogram=histogram)]

    def inputs(self, metadata):
        return [*self.columns, Column("Categories"), Column("Weights")]

    def outputs(self, metadata):
        return []


def makeHistogram(
    product_name, columns, axes, data, description, multirun_strategy=None, mask=None
):
    if not isinstance(data, (list, tuple)):
        data = [data]
        axes = [axes]

    names = []
    for i, d in enumerate(data):
        name = Column(f"INTERNAL_USE.auto-col-{product_name}-{i}")
        names.append(name)
        columns[name] = d

    if mask is not None:
        mask_col_name = Column(f"INTERNAL_USE.mask-{product_name}-{i}")
        columns[make_col_name] = mask
    else:
        mask_col_name = None

    b = HistogramBuilder(product_name, names, axes, axes, mask_col=mask_col_name)
    return ModuleAddition(b, buildVariations)
