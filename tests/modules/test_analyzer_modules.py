import pytest
import awkward as ak
import numpy as np
from analyzer.core.columns import Column
from analyzer.modules.common.jets import (
    JetFilter,
    FilterNear,
    Count,
    PromoteIndex,
    HT,
)
from tests.modules.base_module_test import BaseModuleTest
from tests.modules.fixtures import (
    createMockTrackedColumns,
    createMockMetadata,
    assertColumnExists,
)


class TestJetFilter(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return JetFilter(
            input_col=Column("Jet"),
            output_col=Column("GoodJet"),
            min_pt=30.0,
            max_abs_eta=2.4,
            include_jet_id=True,
            include_pu_id=False,
        )

    def testInputsOutputs(self, module, mockMetadata):
        self.assertInputsCorrect(module, mockMetadata)
        self.assertOutputsCorrect(module, mockMetadata)

        inputs = module.inputs(mockMetadata)
        assert Column("Jet") in inputs

        outputs = module.outputs(mockMetadata)
        assert Column("GoodJet") in outputs

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testFilteringLogic(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        assertColumnExists(output_columns, "GoodJet")
        input_jets = mockColumns["Jet"]
        output_jets = output_columns["GoodJet"]
        assert ak.all(output_jets.pt > 30.0), "Some jets have pt <= 30.0"
        assert ak.all(abs(output_jets.eta) < 2.4), "Some jets have |eta| >= 2.4"
        assert ak.all(ak.num(output_jets) <= ak.num(input_jets)), (
            "Output has more jets than input"
        )

    def testCaching(self, module, mockColumns):
        self.assertCachingWorks(module, mockColumns)

    def testDifferentParameters(self):
        module = JetFilter(
            input_col=Column("Jet"),
            output_col=Column("LooseJet"),
            min_pt=20.0,
            max_abs_eta=5.0,
            include_jet_id=False,
        )

        columns = createMockTrackedColumns()
        output_columns, _ = self.runModule(module, columns)

        output_jets = output_columns["LooseJet"]
        assert ak.all(output_jets.pt > 20.0)
        assert ak.all(abs(output_jets.eta) < 5.0)


class TestCount(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return Count(input_col=Column("Jet"), output_col=Column("nJet"))

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testCountLogic(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        assertColumnExists(output_columns, "nJet")
        input_jets = mockColumns["Jet"]
        output_counts = output_columns["nJet"]
        expected_counts = ak.num(input_jets, axis=1)
        assert ak.all(output_counts == expected_counts), (
            "Counts do not match expected values"
        )
        assert output_counts.ndim == 1, "Output should be 1D array"


class TestPromoteIndex(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return PromoteIndex(
            input_col=Column("Jet"), output_col=Column("LeadingJet"), index=0
        )

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testPromotionLogic(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        assertColumnExists(output_columns, "LeadingJet")
        input_jets = mockColumns["Jet"]
        leading_jet = output_columns["LeadingJet"]
        mask = ak.num(input_jets) > 0
        assert ak.all(leading_jet.pt[mask] == input_jets[:, 0].pt[mask]), (
            "Leading jet pt does not match"
        )

class TestHT(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return HT(input_col=Column("Jet"), output_col=Column("HT"))

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testHTCalculation(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        assertColumnExists(output_columns, "HT")
        input_jets = mockColumns["Jet"]
        ht = output_columns["HT"]
        expected_ht = ak.sum(input_jets.pt, axis=1)
        assert ak.all(ak.isclose(ht, expected_ht)), "HT does not match sum of jet pt"
        assert ht.ndim == 1, "HT should be 1D array"
        assert ak.all(ht >= 0), "HT should be non-negative"


class TestFilterNear(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return FilterNear(
            target_col=Column("Jet"),
            near_col=Column("Muon"),
            output_col=Column("CleanedJet"),
            max_dr=0.4,
        )

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testFilteringReducesJets(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        assertColumnExists(output_columns, "CleanedJet")
        input_jets = mockColumns["Jet"]
        output_jets = output_columns["CleanedJet"]
        assert ak.all(ak.num(output_jets) <= ak.num(input_jets)), (
            "Filtering should not increase jet count"
        )

    def testInputsOutputs(self, module, mockMetadata):
        inputs = module.inputs(mockMetadata)
        assert Column("Jet") in inputs
        assert Column("Muon") in inputs

        outputs = module.outputs(mockMetadata)
        assert Column("CleanedJet") in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
