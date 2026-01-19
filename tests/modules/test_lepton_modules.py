import pytest
import awkward as ak
from analyzer.core.columns import Column
from analyzer.modules.common.muons import MuonMaker, IdWps, IsoWps
from analyzer.modules.common.electrons import ElectronMaker, CutBasedWPs
from tests.modules.base_module_test import BaseModuleTest
from tests.modules.fixtures import assertColumnExists


class TestMuonMaker(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return MuonMaker(
            input_col=Column("Muon"),
            output_col=Column("GoodMuon"),
            id_working_point=IdWps.loose,
            min_pt=25.0,
            max_abs_eta=2.4,
            iso_working_point=IsoWps.loose,
        )

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testFilteringLogic(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        # assertColumnExists(output_columns, "GoodMuon")
        # Check column existence manualy to avoid fixture issues or helper issues
        assert "GoodMuon" in output_columns.fields

        input_muons = mockColumns["Muon"]
        output_muons = output_columns["GoodMuon"]
        assert ak.all(output_muons.pt > 25.0), "Some muons have pt <= 25.0"
        assert ak.all(abs(output_muons.eta) < 2.4), "Some muons have |eta| >= 2.4"
        assert ak.all(ak.num(output_muons) <= ak.num(input_muons))

    def testInputsOutputs(self, module, mockMetadata):
        self.assertInputsCorrect(module, mockMetadata)
        self.assertOutputsCorrect(module, mockMetadata)
        inputs = module.inputs(mockMetadata)
        assert Column("Muon") in inputs
        outputs = module.outputs(mockMetadata)
        assert Column("GoodMuon") in outputs


class TestElectronMaker(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return ElectronMaker(
            input_col=Column("Electron"),
            output_col=Column("GoodElectron"),
            working_point=CutBasedWPs.medium,
            min_pt=30.0,
            max_abs_eta=2.5,
        )

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testFilteringLogic(self, module, mockColumns):
        output_columns, _ = self.runModule(module, mockColumns)
        assert "GoodElectron" in output_columns.fields
        input_electrons = mockColumns["Electron"]
        output_electrons = output_columns["GoodElectron"]
        assert ak.all(output_electrons.pt > 30.0)
        assert ak.all(abs(output_electrons.eta) < 2.5)
        assert ak.all(ak.num(output_electrons) <= ak.num(input_electrons))

    def testInputsOutputs(self, module, mockMetadata):
        self.assertInputsCorrect(module, mockMetadata)
        self.assertOutputsCorrect(module, mockMetadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
