from attrs import define
from analyzer.core.columns import Column
from analyzer.core.analysis_modules import AnalyzerModule
from .axis import Axis, RegularAxis


def addCategory(columns, name, data, axis):
    col = Column(fields=("Categories", name))
    columns[col] = data
    to_add = CategoryDesc(column=col, axis=axis)
    if "categories" not in columns.pipeline_data:
        columns.pipeline_data["categories"] = []
    columns.pipeline_data["categories"].append(to_add)
    return columns


@define(frozen=True)
class CategoryDesc:
    column: Column
    axis: Axis


@define
class SimpleCategory(AnalyzerModule):
    """
    Create a simple numerical category from an input column.

    This analyzer maps a numeric column to a category and registers it
    in the pipeline with associated axis information for plotting or
    binning.
    This axis will be added to all subsequent histograms.

    Parameters
    ----------
    input_col : Column
        Input column whose values define the category.
    cat_name : str
        Name of the category to create.
    bins : int
        Number of bins for the category axis.
    start : float
        Minimum value of the category axis.
    stop : float
        Maximum value of the category axis.
    """

    input_col: Column
    cat_name: str
    bins: int
    start: float
    stop: float

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(fields=("Categories", self.cat_name))]

    def run(self, columns, params):
        col = columns[self.input_col]
        addCategory(
            columns,
            self.cat_name,
            col,
            RegularAxis(
                bins=self.bins, start=self.start, stop=self.stop, name=self.cat_name
            ),
        )
        return columns, []
