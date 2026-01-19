import pytest
from pathlib import Path
from analyzer.core.event_collection import FileChunk, getFileInfo
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column, TrackedColumns


def testRunSimpleAnalysis():
    # Resolve path to test data
    base_dir = Path(__file__).parent
    data_file = base_dir / "test_data" / "nano_dy.root"

    assert data_file.exists(), f"Test data file not found at {data_file}"

    # 1. Get file info
    # We use tree_name="Events" as is standard for NanoAOD
    file_info = getFileInfo(str(data_file), "Events")
    assert file_info.nevents > 0

    # 2. Create a FileChunk for the entire file
    chunk = FileChunk(
        file_path=str(data_file),
        event_start=0,
        event_stop=file_info.nevents,
        tree_name="Events",
        file_nevents=file_info.nevents,
    )

    # 3. Load events
    # We use coffea-virtual backend for testing
    columns = chunk.loadEvents(
        backend="coffea-virtual", view_kwargs={"metadata": {}, "provenance": 0}
    )
    assert isinstance(columns, TrackedColumns)

    # 4. Define a simple AnalyzerModule
    class MuonPtAnalyzer(AnalyzerModule):
        def inputs(self, metadata):
            return [Column("Muon.pt")]

        def outputs(self, metadata):
            return [Column("MuonPtSum")]

        def run(self, columns, params):
            # Simple logic: sum of muon pts per event
            import awkward as ak

            muons = columns["Muon"]
            pt_sum = ak.sum(muons.pt, axis=-1)
            return [], [pt_sum]

    # 5. Run the module
    module = MuonPtAnalyzer()
    # Mock metadata if needed, though TrackedColumns likely has it initialized to None or empty
    # check default metadata

    # AnalyzerModule.run returns (updated_columns, results)
    # But usually we call module(columns, params)

    # We need to make sure specific columns are accessible.
    # Note: TrackedColumns from loadEvents wraps the nanoevents.
    # We might need to ensure "Muon.pt" is accessible via the column interface used in AnalyzerModule.

    # Let's try running it.
    res_cols, results = module(columns, {})

    # 6. Verify results
    # results[0] should be the pt_sum array
    assert len(results) == 1
    pt_sums = results[0]

    assert len(pt_sums) == file_info.nevents
    # Check that we have some non-zero values (assuming there are muons in the file)
    # It's Drell-Yan, so there should be muons.
    import awkward as ak

    assert ak.any(pt_sums > 0)
