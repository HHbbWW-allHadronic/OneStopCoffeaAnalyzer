import pytest
import awkward as ak
import numpy as np
from analyzer.core.columns import Column
from analyzer.modules.common.histogram_builder import (
    SimpleHistogram,
    HistogramBuilder,
    makeHistogram,
)
from analyzer.modules.common.axis import RegularAxis
from tests.modules.base_module_test import BaseModuleTest
from analyzer.core.results import Histogram


class TestSimpleHistogram(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return SimpleHistogram(
            hist_name="pt_hist",
            input_cols=[Column("Jet") + "pt"],
            axes=[RegularAxis(50, 0, 500, "pt", "Jet pT")],
        )

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testHistogramCreation(self, module, mockColumns):
        output_columns, results = self.runModule(module, mockColumns)

        # Check that results contain the histogram builder addition
        assert len(results) == 1
        addition = results[0]
        assert hasattr(addition, "analyzer_module")
        builder = addition.analyzer_module
        assert isinstance(builder, HistogramBuilder)
        assert builder.product_name == "pt_hist"

    def testInputs(self, module, mockMetadata):
        inputs = module.inputs(mockMetadata)
        assert len(inputs) == 1
        assert inputs[0] == Column("Jet") + "pt"


class TestHistogramBuilder(BaseModuleTest):
    def testFillTransform(self):
        # Test 1D array
        data = ak.Array([1, 2, 3])
        res = HistogramBuilder.maybeFlatten(data)
        assert ak.all(res == data)

        # Test 2D array
        data2d = ak.Array([[1, 2], [3]])
        res2d = HistogramBuilder.maybeFlatten(data2d)
        assert len(res2d) == 3
        assert ak.all(res2d == ak.Array([1, 2, 3]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
