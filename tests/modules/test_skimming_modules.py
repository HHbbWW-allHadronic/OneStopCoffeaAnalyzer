import pytest
import awkward as ak
from analyzer.modules.common.skimming import SaveEvents
from tests.modules.base_module_test import BaseModuleTest
from unittest.mock import MagicMock, patch
from analyzer.core.columns import Column
import pathlib
import shutil
import uproot


class TestSaveEvents(BaseModuleTest):
    @pytest.fixture
    def module(self):
        return SaveEvents(
            prefix="test_prefix_", output_format="{dataset_name}_{file_id}.root"
        )

    def testModuleRuns(self, module, mockColumns):
        with (
            patch("analyzer.modules.common.skimming.uproot") as mock_uproot,
            patch("analyzer.modules.common.skimming.copyFile") as mock_copy,
            patch("analyzer.modules.common.skimming.Path") as mock_path_cls,
        ):
            mock_base_path = MagicMock()
            mock_local_file = MagicMock()
            mock_path_cls.return_value = mock_base_path
            mock_base_path.__truediv__.return_value = mock_local_file

            mock_file_handle = MagicMock()
            mock_uproot.recreate.return_value.__enter__.return_value = mock_file_handle

            self.assertModuleRunsWithoutError(module, mockColumns)

            mock_base_path.mkdir.assert_called()
            mock_uproot.recreate.assert_called()
            mock_copy.assert_called()
            mock_local_file.unlink.assert_called()

    def testOutputNaming(self, module, mockColumns):
        with (
            patch("analyzer.modules.common.skimming.uproot"),
            patch("analyzer.modules.common.skimming.copyFile") as mock_copy,
            patch("analyzer.modules.common.skimming.Path") as mock_path_cls,
        ):
            mock_path_cls.return_value = MagicMock()

            self.runModule(module, mockColumns)

            mock_copy.assert_called()
            args, _ = mock_copy.call_args
            target_path = args[1]

            assert target_path.startswith("test_prefix_test_dataset_")

    def testUprootWriting(self, module, mockColumns, tmp_path):
        module.prefix = str(tmp_path / "final_")

        real_Path = pathlib.Path

        def path_side_effect(*args, **kwargs):
            if args and args[0] == "localsaved":
                return tmp_path / "localsaved"
            return real_Path(*args, **kwargs)

        def copy_side_effect(src, dst):
            shutil.copy(src, dst)

        with (
            patch(
                "analyzer.modules.common.skimming.Path", side_effect=path_side_effect
            ),
            patch(
                "analyzer.modules.common.skimming.copyFile",
                side_effect=copy_side_effect,
            ),
        ):
            output_columns, _ = self.runModule(module, mockColumns)

            final_files = list(tmp_path.glob("final_*.root"))
            assert len(final_files) == 1, (
                f"Expected 1 output file, found {len(final_files)}. Files: {list(tmp_path.glob('*'))}"
            )

            output_file = final_files[0]

            with uproot.open(output_file) as f:
                assert "Events" in f
                events_tree = f["Events"]
                keys = events_tree.keys()
                assert len(keys) > 0
                assert "event" in keys

                assert events_tree.num_entries == len(mockColumns["event"])

                saved_events = events_tree["event"].array()
                assert ak.all(saved_events == mockColumns["event"])

    def testInputsOutputs(self, module, mockMetadata):
        inputs = module.inputs(mockMetadata)
        assert inputs == "EVENTS"

        outputs = module.outputs(mockMetadata)
        assert outputs == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
