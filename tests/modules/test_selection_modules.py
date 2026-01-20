import pytest
import awkward as ak
from analyzer.core.columns import Column
from analyzer.modules.common.selection import SelectOnColumns, NObjFilter
from analyzer.modules.common.hlt_selection import SimpleHLT
from tests.modules.base_module_test import BaseModuleTest
from tests.modules.fixtures import assertColumnExists


class TestSelectOnColumns(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return SelectOnColumns(
            sel_name="test_selection_summary",
            selection_names=["filter1", "filter2"],
        )

    def testModuleRuns(self, module, mockColumns):
        # Setup pipeline data with selections
        if "Selections" not in mockColumns.pipeline_data:
            mockColumns.pipeline_data["Selections"] = {}

        n_events = len(mockColumns["event"])
        mockColumns["Selection", "filter1"] = ak.Array([True] * n_events)
        mockColumns["Selection", "filter2"] = ak.Array([True] * n_events)

        self.assertModuleRunsWithoutError(module, mockColumns)

    def testSelectionLogic(self, module, mockColumns):
        if "Selections" not in mockColumns.pipeline_data:
            mockColumns.pipeline_data["Selections"] = {}

        n_events = 3
        # Manually create events to control length
        from tests.modules.fixtures import createMockEvents, createMockTrackedColumns

        events = createMockEvents(n_events=n_events)
        columns = createMockTrackedColumns(events)
        columns.pipeline_data["Selections"] = {}

        # filter1: T, T, F
        # filter2: T, F, T
        # Result:  T, F, F
        columns["Selection", "filter1"] = ak.Array([True, True, False])
        columns["Selection", "filter2"] = ak.Array([True, False, True])

        output_columns, results = self.runModule(module, columns)

        # Check filtered events (only index 0 remains)
        assert len(output_columns["event"]) == 1

        # Check cutflow result
        assert len(results) == 1
        cutflow = results[0]
        assert cutflow.name == "test_selection_summary"
        assert cutflow.cutflow["initial"] == 3
        assert cutflow.cutflow["filter1"] == 2
        assert cutflow.cutflow["filter2"] == 1


class TestNObjFilter(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return NObjFilter(
            selection_name="min_jets",
            input_col=Column("Jet"),
            min_count=2,
            max_count=None,
        )

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testFiltering(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        # Fix: assertColumnExists takes Column or str.
        # "Selection", "min_jets" implies Column(("Selection", "min_jets"))
        assertColumnExists(output_columns, Column(("Selection", "min_jets")))

        selection = output_columns["Selection", "min_jets"]
        jets = mockColumns["Jet"]

        # Check logic: nJet >= 2
        expected = ak.num(jets) >= 2
        assert ak.all(selection == expected)


class TestSimpleHLT(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return SimpleHLT(triggers=["HT"], selection_name="PassTrigger")

    def testModuleRuns(self, module, mockColumns):
        # Add trigger bit
        n_events = len(mockColumns["event"])
        mockColumns["HLT", "PFHT1050"] = ak.Array([True] * n_events)
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testInputsOutputs(self, module, mockMetadata):
        # We need mock metadata to have the trigger name mapping
        # createMockMetadata provides "HT": "PFHT1050"
        self.assertInputsCorrect(module, mockMetadata)

        outputs = module.outputs(mockMetadata)
        # Fix: Column creation with fields
        target = Column(("Selection", "PassTrigger"))
        assert target in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
