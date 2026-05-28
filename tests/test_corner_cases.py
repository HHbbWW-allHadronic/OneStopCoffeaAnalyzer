import pytest
import yaml
import awkward as ak
from pathlib import Path
from analyzer.core.running import runFromPath
from analyzer.core.results import loadResults
from analyzer.core.analysis_modules import AnalyzerModule, Column

@pytest.fixture
def corner_cases_setup(tmp_path):
    # Setup directory structure
    config_dir = tmp_path / "config"
    datasets_dir = tmp_path / "datasets"
    eras_dir = tmp_path / "eras"
    output_dir = tmp_path / "output"

    config_dir.mkdir()
    datasets_dir.mkdir()
    eras_dir.mkdir()
    output_dir.mkdir()

    base_dir = Path(__file__).parent.parent
    
    # We'll use a mock dataset for empty events
    era_def = [{"name": "2018_test", "luminosity": 1.0}]
    with open(eras_dir / "2018_test.yaml", "w") as f:
        yaml.dump(era_def, f)

    return config_dir, datasets_dir, eras_dir, output_dir, base_dir

def testEmptyEvents(corner_cases_setup):
    config_dir, datasets_dir, eras_dir, output_dir, base_dir = corner_cases_setup
    
    # Path to a file with 0 events or just use a small file and max_sample_events=0
    # Actually, we can't easily create a 0-event ROOT file here without uproot.
    # We'll use a small file and filter everything out.
    
    data_file = base_dir / "tests" / "test_data" / "nano_dy.root"
    
    dataset_def = [{
        "name": "empty_test",
        "title": "Empty Test Dataset",
        "era": "2018_test",
        "sample_type": "MC",
        "samples": [{
            "name": "empty_sample",
            "n_events": 100,
            "x_sec": 1.0,
            "source": {
                "files": [str(data_file.absolute())],
                "type": "FileListCollection",
                "tree_name": "Events",
            },
        }],
    }]
    with open(datasets_dir / "empty_test.yaml", "w") as f:
        yaml.dump(dataset_def, f)

    # A module that filters out all events
    module_file = config_dir / "filtering_module.py"
    with open(module_file, "w") as f:
        f.write("""
from analyzer.core.analysis_modules import AnalyzerModule
from analyzer.core.columns import Column
from attrs import define
import awkward as ak

@define
class FilterAll(AnalyzerModule):
    def run(self, columns, params):
        mask = ak.zeros_like(columns["Muon"].pt, dtype=bool)
        columns["Muon"] = columns["Muon"][mask]
        return columns, []
    def inputs(self, metadata): return [Column("Muon")]
    def outputs(self, metadata): return [Column("Muon")]
""")

    analysis_config = {
        "analyzer": {
            "nominal": [
                {"module_name": "FilterAll"},
                {
                    "module_name": "JetComboHistograms",
                    "input_col": "Muon",
                    "prefix": "empty",
                    "jet_combos": [[0, 1]],
                },
            ],
        },
        "event_collections": [{"dataset": "empty_test", "pipelines": ["nominal"]}],
        "extra_dataset_paths": [str(datasets_dir)],
        "extra_era_paths": [str(eras_dir)],
        "extra_module_paths": [str(module_file)],
    }
    
    analysis_file = config_dir / "analysis.yaml"
    with open(analysis_file, "w") as f:
        yaml.dump(analysis_config, f)

    # Should run without crashing even if 0 events remain
    runFromPath(str(analysis_file), str(output_dir), executor_name="imm-testing")
    
    expected_output = output_dir / "empty_test__empty_sample.result"
    assert expected_output.exists()
    
    results = loadResults([str(expected_output)], peek_only=False)
    hist = results["empty_test"]["empty_sample"]["pipelines"]["nominal"]["empty_12_m"].histogram
    assert hist.sum().value == 0
