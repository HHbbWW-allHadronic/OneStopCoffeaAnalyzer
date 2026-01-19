import pytest
import yaml
from pathlib import Path
from analyzer.core.running import runFromPath
from analyzer.core.results import loadResults
import analyzer.modules.common.jets  # noqa: F401


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

    # Path to Custom Systematics Modules
    syst_module_file = base_dir / "tests" / "modules" / "syst_modules.py"
    assert syst_module_file.exists(), (
        f"Systematics module file not found at {syst_module_file}"
    )

    # Standard Module Config
    hist_mod_cfg = {
        "module_name": "JetComboHistograms",
        "input_col": "Muon",
        "prefix": "di_muon",
        "jet_combos": [[0, 1]],
        "mass_axis": {
            "bins": 100,
            "start": 50,
            "stop": 150,
            "unit": "GeV",
        },
    }

    # Add TestCachingModule to pipeline
    caching_mod_cfg = {
        "module_name": "TestCachingModule",
        "configuration": {},
    }

    # Create Analysis Configuration
    analysis_config = {
        "analyzer": {
            "nominal": [
                {"module_name": "MuonScaleSyst", "configuration": {}},
                {"module_name": "MuonResSyst", "configuration": {}},
                {"module_name": "EventWeightSyst", "configuration": {}},
                caching_mod_cfg,  # Added here
                hist_mod_cfg,
            ],
        },
        "event_collections": [
            {
                "dataset": "dy_test",
                "pipelines": [
                    "nominal",
                ],
            }
        ],
        "extra_dataset_paths": [str(datasets_dir)],
        "extra_era_paths": [str(eras_dir)],
        "extra_module_paths": [str(syst_module_file)],
        "extra_executors": {},
        "nominal": [hist_mod_cfg],
    }

    analysis_file = config_dir / "analysis.yaml"
    with open(analysis_file, "w") as f:
        yaml.dump(analysis_config, f)

    return analysis_file, output_dir


@pytest.mark.filterwarnings("ignore::RuntimeWarning:coffea.*")
def test_run_e2e_analysis(e2e_setup):
    import numpy as np

    config_path, output_dir = e2e_setup

    analyzer = runFromPath(
        str(config_path),
        str(output_dir),
        executor_name="imm-testing",
        max_sample_events=None,
        return_analyzer=True,
    )

    caching_module = None

    found_caching_modules = [
        m for m in analyzer.all_modules if m.__class__.__name__ == "TestCachingModule"
    ]

    # Assert we found it
    assert len(found_caching_modules) > 0, "TestCachingModule not found in analyzer"
    caching_module = found_caching_modules[0]

    counts = len(caching_module.execution_counts)
    print(f"TestCachingModule execution count: {counts}")
    # We debug the actual params and means if needed
    for i, ctx in enumerate(caching_module.execution_counts):
        print(f"  Run {i + 1}: pt_mean={ctx['pt_mean']:.4f} params={ctx['params']}")

    assert counts == 4, (
        f"Expected 4 executions (Nom, ScaleUp, ScaleDown, ResUp), got {counts}. Weight syst should hit cache."
    )
    print(
        "Integrated Module Caching Verified: Weight systematic correctly reused nominal result."
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

    # Debug: Print available variations
    if len(hist_mass.axes) > 0 and hist_mass.axes[0].name == "variation":
        print(f"Available variations: {list(hist_mass.axes[0])}")

    def get_stats(hist_obj, variation_name):
        # Slice variation
        try:
            h_var = hist_obj[{"variation": variation_name}]
        except KeyError:
            print(f"Variation {variation_name} not found in histogram")
            return None

        counts = h_var.values()
        centers = h_var.axes[0].centers
        total = float(np.sum(counts))

        if total > 0:
            mean = np.average(centers, weights=counts)
            count_cumsum = np.cumsum(counts)
            idx = np.searchsorted(count_cumsum, count_cumsum[-1] / 2)
            median = centers[idx]
            std_dev = np.sqrt(np.average((centers - mean) ** 2, weights=counts))
            return {"total": total, "mean": mean, "median": median, "std_dev": std_dev}
        return None

    # Get Nominal Stats
    stats_nom = get_stats(hist_mass, "central")
    assert stats_nom is not None
    print(
        f"\nNominal: Mean={stats_nom['mean']:.2f}, Median={stats_nom['median']:.2f}, Std={stats_nom['std_dev']:.2f}, Yield={stats_nom['total']}"
    )

    # Check Z Mass
    z_mass = 91.1876
    assert abs(stats_nom["median"] - z_mass) < 10.0, (
        f"Nominal median {stats_nom['median']} not close to Z mass"
    )

    # Scale Systematics
    stats_up = get_stats(hist_mass, "MuonScaleSyst_variation_up")
    stats_down = get_stats(hist_mass, "MuonScaleSyst_variation_down")

    assert stats_up is not None
    assert stats_down is not None

    print(f"Scale Up: Median={stats_up['median']:.2f}")
    print(f"Scale Down: Median={stats_down['median']:.2f}")

    # Assert mass shift
    assert stats_up["median"] > stats_nom["median"], (
        "Scale Up should increase median mass"
    )
    assert stats_down["median"] < stats_nom["median"], (
        "Scale Down should decrease median mass"
    )

    # Resolution Systematics
    stats_res = get_stats(hist_mass, "MuonResSyst_variation_up")

    assert stats_res is not None
    print(f"Res Up: Std={stats_res['std_dev']:.2f}")

    # Assert resolution smear check
    assert stats_res["std_dev"] > stats_nom["std_dev"], (
        "Resolution smear should increase width"
    )

    # Weight Systematics
    stats_weight = get_stats(hist_mass, "EventWeightSyst_variation_up")

    assert stats_weight is not None
    print(f"Weight Up: Yield={stats_weight['total']}")

    # Assert yield check
    assert stats_weight["total"] > stats_nom["total"], "Weight Up should increase yield"
    expected = stats_nom["total"] * 1.5
    assert abs(stats_weight["total"] - expected) < 1.0, (
        f"Weight yield {stats_weight['total']} mismatch expected {expected}"
    )
