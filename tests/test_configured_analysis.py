import pytest
import yaml
from pathlib import Path
from analyzer.core.running import runFromPath
from analyzer.core.results import loadResults
import analyzer.modules.common.jets  # Ensure module is registered


@pytest.fixture
def e2e_setup(tmp_path):
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
            "luminosity": 1.0,  # Dummy
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

    analysis_config = {
        "analyzer": {
            "nominal": [
                {
                    "module_name": "JetComboHistograms",
                    "input_col": "Muon",
                    "prefix": "di_muon",
                    "jet_combos": [[0, 1]],
                }
            ]
        },
        "event_collections": [{"dataset": "dy_test", "pipelines": ["nominal"]}],
        "extra_dataset_paths": [str(datasets_dir)],
        "extra_era_paths": [str(eras_dir)],
        "extra_module_paths": [],
        "extra_executors": {},
        "nominal": [
            {
                "module_name": "JetComboHistograms",
                "input_col": "Muon",
                "prefix": "di_muon",
                "jet_combos": [[0, 1]],
            }
        ],
    }

    analysis_file = config_dir / "analysis.yaml"
    with open(analysis_file, "w") as f:
        yaml.dump(analysis_config, f)

    return analysis_file, output_dir


@pytest.mark.filterwarnings("ignore::RuntimeWarning:coffea.*")
def test_run_e2e_analysis(e2e_setup):
    import hist
    import numpy as np

    config_path, output_dir = e2e_setup

    runFromPath(
        str(config_path),
        str(output_dir),
        executor_name="imm-1000",
        max_sample_events=None,
    )
    expected_output = output_dir / "dy_test__dy_sample.result"
    assert expected_output.exists()

    results = loadResults([str(expected_output)], peek_only=False)

    assert results.name == "ROOT"
    dataset_res = results["dy_test"]
    assert dataset_res is not None
    sample_res = dataset_res["dy_sample"]
    assert sample_res is not None
    pipeline_res = sample_res["pipelines"]
    assert pipeline_res is not None
    nominal_res = pipeline_res["nominal"]
    assert nominal_res is not None

    hist_mass = nominal_res["di_muon_12_m"].histogram
    assert hist_mass is not None

    h_central = hist_mass[{"variation": "central"}]
    counts = h_central.values()
    centers = h_central.axes[0].centers
    mean = np.average(centers, weights=counts)
    count_cumsum = np.cumsum(counts)
    idx = np.searchsorted(count_cumsum, count_cumsum[-1] / 2)
    median = centers[idx]
    std_dev = np.sqrt(np.average((centers - mean) ** 2, weights=counts))

    print(f"Mean: {mean:.2f} GeV")
    print(f"Median: {median:.2f} GeV")
    print(f"Std Dev: {std_dev:.2f} GeV")

    z_mass = 91.1876
    assert abs(median - z_mass) < 5.0, (
        f"Median mass {median} is not close enough to Z mass {z_mass}"
    )
