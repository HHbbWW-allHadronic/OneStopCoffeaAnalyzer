import pytest
import awkward as ak
from analyzer.core.columns import Column
from analyzer.modules.common.categories import SimpleCategory
from tests.modules.base_module_test import BaseModuleTest


class TestSimpleCategory(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return SimpleCategory(
            input_col=Column("Jet") + "pt", cat_name="qt", bins=50, start=0, stop=500
        )

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testCategoryCreated(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        assert "Categories" in output_columns.fields
        assert "qt" in output_columns["Categories"].fields

    def testInputsOutputs(self, module, mockMetadata):
        self.assertInputsCorrect(module, mockMetadata)
        self.assertOutputsCorrect(module, mockMetadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
