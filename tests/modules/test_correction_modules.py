import pytest
import awkward as ak
import numpy as np
from analyzer.core.columns import Column
from analyzer.modules.common.event_level_corrections import (
    PosNegGenWeight,
    NoiseFilter,
    GoldenLumi,
    PileupSF,
    L1PrefiringSF,
)
from analyzer.modules.common.bjet_sf import BJetShapeSF
from tests.modules.base_module_test import BaseModuleTest
from unittest.mock import MagicMock, patch
import correctionlib


class TestPosNegGenWeight(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return PosNegGenWeight()

    def testModuleRuns(self, module, mockColumns):
        n = len(mockColumns["event"])
        mockColumns["genWeight"] = ak.full_like(mockColumns["event"], 1.0, dtype=float)
        self.assertModuleRunsWithoutError(module, mockColumns)

    def testInputsOutputs(self, module, mockMetadata):
        self.assertInputsCorrect(module, mockMetadata)
        self.assertOutputsCorrect(module, mockMetadata)


class TestNoiseFilter(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return NoiseFilter()

    def testModuleRuns(self, module, mockColumns):
        self.assertModuleRunsWithoutError(module, mockColumns)
