import pytest
import yaml
import awkward as ak
import numpy as np
from pathlib import Path
from analyzer.core.running import runFromPath
from analyzer.core.results import loadResults
from tests.modules.advanced_test_modules import ExecutionCounter

@pytest.fixture
def advanced_e2e_setup(tmp_path):
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
    data_file = base_dir / "tests" / "test_data" / "nano_dy.root"
    assert data_file.exists(), "Test data file not found"

    era_def = [
        {
            "name": "2018_test",
            "luminosity": 1.0,
        }
    ]
    with open(eras_dir / "2018_test.yaml", "w") as f:
        yaml.dump(era_def, f)

    dataset_def = [
        {
            "name": "dy_test",
            "title": "DY Test Dataset",
            "era": "2018_test",
            "sample_type": "MC",
            "samples": [
                {
                    "name": "dy_sample",
                    "n_events": 100,
                    "x_sec": 1.0,
                    "source": {
                        "files": [str(data_file.absolute())],
                        "type": "FileListCollection",
                        "tree_name": "Events",
                    },
                }
            ],
        }
    ]
    with open(datasets_dir / "dy_test.yaml", "w") as f:
        yaml.dump(dataset_def, f)

    # Path to Advanced Test Modules
    module_file = base_dir / "tests" / "modules" / "advanced_test_modules.py"
    
    analysis_config = {
        "analyzer": {
            "default_run_builder": {"strategy_name": "CompleteSysts"},
            "nominal": [
                {"module_name": "ExecutionCounter", "counter_name": "start"},
                {"module_name": "ObjectProducer", "input_col": "Muon", "output_col": "ProducedMuon"},
                {"module_name": "ExecutionCounter", "counter_name": "after_producer"},
                {"module_name": "DependentModule", "input_col": "ProducedMuon", "output_col": "DependentPT"},
                {"module_name": "CorrelatedSystA"},
                {"module_name": "CorrelatedSystB"},
                {"module_name": "IndependentWeightSyst"},
                {
                    "module_name": "JetComboHistograms",
                    "input_col": "Muon",
                    "prefix": "muon_mass",
                    "jet_combos": [[0, 1]],
                    "mass_axis": {"bins": 100, "start": 50, "stop": 150, "unit": "GeV"},
                },
            ],
        },
        "event_collections": [
            {
                "dataset": "dy_test",
                "pipelines": ["nominal"],
            }
        ],
        "extra_dataset_paths": [str(datasets_dir)],
        "extra_era_paths": [str(eras_dir)],
        "extra_module_paths": [str(module_file)],
    }

    analysis_file = config_dir / "analysis.yaml"
    with open(analysis_file, "w") as f:
        yaml.dump(analysis_config, f)

    return analysis_file, output_dir
    
def testAdvancedE2E(advanced_e2e_setup):
    config_path, output_dir = advanced_e2e_setup
    
    # Reset counts
    ExecutionCounter.counts = {}

    analyzer = runFromPath(
        str(config_path),
        str(output_dir),
        executor_name="imm-testing",
        return_analyzer=True,
    )

    # --- Deep Analysis Assertions ---
    from analyzer.core.running import getRepos

    # 1. Analyzer Configuration State
    # Modules: LoadColumns + 2 Counters + ObjectProducer + DependentModule + 2 CorrelatedSysts + IndependentWeightSyst + JetComboHistograms = 9
    assert len(analyzer.all_modules) >= 8
    assert analyzer.default_run_builder.__class__.__name__ == "CompleteSysts"
    
    # Verify module parameters were correctly deserialized
    producer = next(m for m in analyzer.all_modules if m.__class__.__name__ == "ObjectProducer")
    assert str(producer.input_col) == "Muon"
    assert str(producer.output_col) == "ProducedMuon"

    # 2. Dataset and Era Repository Integrity
    # We reload them to verify they match what we set up
    tmp_path = advanced_e2e_setup[0].parent.parent
    dataset_repo, era_repo = getRepos([str(tmp_path / "datasets")], [str(tmp_path / "eras")])
    
    assert "dy_test" in dataset_repo.datasets
    assert "2018_test" in era_repo.eras
    dataset = dataset_repo["dy_test"]
    assert dataset.era == "2018_test"
    assert len(dataset.samples) == 1
    assert dataset.samples[0].name == "dy_sample"

    # 3. Module Dependency Verification (Inputs/Outputs)
    dependent = next(m for m in analyzer.all_modules if m.__class__.__name__ == "DependentModule")
    # Verify it correctly identifies its input/output columns
    # We pass a dummy metadata dict
    dummy_meta = {"sample_type": "MC"}
    assert any(str(c) == "ProducedMuon" for c in dependent.inputs(dummy_meta))
    assert any(str(c) == "DependentPT" for c in dependent.outputs(dummy_meta))

    # 4. Result Deep Dive
    expected_output = output_dir / "dy_test__dy_sample.result"
    assert expected_output.exists()

    results = loadResults([str(expected_output)], peek_only=False)
    assert "dy_test" in results
    assert "dy_sample" in results["dy_test"]
    
    nominal_res = results["dy_test"]["dy_sample"]["pipelines"]["nominal"]
    
    # Check Histogram Structure
    hist_obj = nominal_res["muon_mass_12_m"]
    hist = hist_obj.histogram
    variations = [str(x) for x in hist.axes[0]]
    print(f"Found variations: {variations}")
    
    assert "central" in variations
    assert any("param_A" in v and "up" in v for v in variations)
    assert any("param_A" in v and "down" in v for v in variations)
    assert any("independent_weight" in v and "up" in v for v in variations)
    assert any("independent_weight" in v and "down" in v for v in variations)

    # 5. Yield Verification (Testing IndependentWeightSyst)
    # Get values for central, independent_weight_up, and independent_weight_down
    # In IndependentWeightSyst: up = 1.1, down = 0.9
    central_yield = hist["central", ...].sum().value
    up_yield = next(hist[v, ...].sum().value for v in variations if "independent_weight" in v and "up" in v)
    down_yield = next(hist[v, ...].sum().value for v in variations if "independent_weight" in v and "down" in v)

    print(f"Yields: central={central_yield}, up={up_yield}, down={down_yield}")
    
    # Assert yields match the weight factors
    assert pytest.approx(up_yield) == central_yield * 1.1
    assert pytest.approx(down_yield) == central_yield * 0.9

    # 6. Execution Trace/Caching Verification
    # start and after_producer should be cached for all variations because they have no inputs or constant inputs.
    start_counter = next(m for m in analyzer.all_modules if getattr(m, "counter_name", None) == "start")
    after_producer_counter = next(m for m in analyzer.all_modules if getattr(m, "counter_name", None) == "after_producer")
    assert start_counter.__class__.counts["start"] == 1
    assert after_producer_counter.__class__.counts["after_producer"] == 1
