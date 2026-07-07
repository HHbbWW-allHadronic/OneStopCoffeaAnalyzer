import yaml
import pytest
import correctionlib
from analyzer.modules.common.jets import JetScaleCorrections
from analyzer.modules.common.bjet_sf import BJetShapeSF
from analyzer.core.datasets import SampleType
from pathlib import Path
from analyzer.core.columns import Column
from analyzer.core.run_builders import CompleteSysts
from analyzer.core.param_specs import ModuleParameterSpec


def load_era_metadata(era_name):
    # Paths are relative to the root of the repo
    era_path = f"analyzer_resources/eras/{era_name}.yaml"
    with open(era_path, "r") as f:
        data = yaml.safe_load(f)
    # The YAML files are lists of dictionaries, usually with one element
    return data[0]


@pytest.mark.skipif(not Path("/cvmfs").exists(), reason="CVMFS Not Available")
@pytest.mark.parametrize("era_name", ["2018", "2017", "2016_preVFP", "2016_postVFP"])
def test_jes_btag_correlation_specs(era_name):
    metadata = {"era": load_era_metadata(era_name), "sample_type": SampleType.MC}

    # Initialize modules
    # JetScaleCorrections needs some columns and options
    jes_mod = JetScaleCorrections(
        input_col=Column("Jets"), output_col=Column("CorrectedJets"), use_regrouped=True
    )

    # BJetShapeSF needs some columns and options
    btag_mod = BJetShapeSF(input_col=Column("CorrectedJets"), weight_name="btagWeight")

    # Get specs
    jes_spec = jes_mod.getParameterSpec(metadata)
    btag_spec = btag_mod.getParameterSpec(metadata)

    assert "jes-variation" in jes_spec
    assert "bjetshapesf-variation" in btag_spec

    jes_param = jes_spec["jes-variation"]
    btag_param = btag_spec["bjetshapesf-variation"]

    # Check that for each JES variation, there is a corresponding btag variation
    # unless it's 'central'
    jes_possible = jes_param.possible_values
    btag_possible = btag_param.possible_values

    driven_by = btag_param.driven_by
    assert "jes-variation" in driven_by

    mapping_func = driven_by["jes-variation"]

    for val in jes_possible:
        if val == "central":
            assert mapping_func(val) is None
        else:
            mapped = mapping_func(val)
            assert mapped is not None
            assert mapped in btag_possible
            # Example: up_jesRegrouped_Absolute -> up_jesAbsolute
            if "Regrouped_" in val:
                assert "Regrouped_" not in mapped
                assert val.replace("Regrouped_", "") == mapped
            else:
                assert val == mapped


@pytest.mark.skipif(not Path("/cvmfs").exists(), reason="CVMFS Not Available")
@pytest.mark.parametrize("era_name", ["2018", "2023_preBPix"])
def test_run_builder_integration(era_name):
    metadata = {
        "era": load_era_metadata(era_name),
        "sample_type": SampleType.MC,
        "dataset_name": "test_dataset",
    }

    jes_mod = JetScaleCorrections(Column("J"), Column("CJ"))
    btag_mod = BJetShapeSF(Column("CJ"), "W")

    jes_spec = jes_mod.getParameterSpec(metadata)
    btag_spec = btag_mod.getParameterSpec(metadata)

    # Merge specs
    full_spec = {}
    full_spec.update(jes_spec)
    full_spec.update(btag_spec)

    builder = CompleteSysts()
    combos = builder(full_spec, metadata)

    # We expect:
    # 1. ("central", {})
    # 2. variations of jes-variation (which should set bjetshapesf-variation)
    # 3. variations of bjetshapesf-variation (independent ones like up_hf, up_lf)

    jes_param = full_spec["jes-variation"]
    btag_param = full_spec["bjetshapesf-variation"]

    jes_variations = [v for v in jes_param.possible_values if v != "central"]
    btag_independent = btag_param.getIndependentValues(full_spec)

    # We expect: jes_variations (which set btag) + independent btag variations (including disabled and central)
    expected_count = len(jes_variations) + len(btag_independent)
    assert len(combos) == expected_count

    # Verify a correlated combo only if it's supposed to be correlated
    jes_correlated = metadata["era"]["btag_scale_factors"].get(
        "jes_correlated_systematics", []
    )
    if jes_correlated:
        regrouped_abs_up = "up_jesRegrouped_Absolute"
        found_correlated = False
        for name, values in combos:
            if values.get("jes-variation") == regrouped_abs_up:
                # Should have mapped to up_jesAbsolute
                assert values.get("bjetshapesf-variation") == "up_jesAbsolute"
                found_correlated = True

        assert found_correlated, f"Could not find combo for {regrouped_abs_up}"

    # Check that we have weight variations too
    found_btag_lf = False
    for name, values in combos:
        if values.get("bjetshapesf-variation") == "up_lf":
            found_btag_lf = True
            # Should NOT have a JES variation set
            assert "jes-variation" not in values or values["jes-variation"] == "central"

    assert found_btag_lf, "Could not find independent btag variation up_lf"


@pytest.mark.skipif(not Path("/cvmfs").exists(), reason="CVMFS Not Available")
@pytest.mark.parametrize(
    "era_name",
    [
        "2018",
        "2017",
        "2016_preVFP",
        "2016_postVFP",
        "2022_preEE",
        "2022_postEE",
        "2023_preBPix",
        "2023_postBPix",
    ],
)
def test_correctionlib_presence(era_name):
    metadata = {"era": load_era_metadata(era_name), "sample_type": SampleType.MC}

    # 1. Check JEC
    jec_params = metadata["era"]["jet_corrections"]
    jec_file = jec_params["files"]["AK4"]
    cset_jec = correctionlib.CorrectionSet.from_file(jec_file)

    # Logic from JetScaleCorrections.getKeyJec
    campaign = jec_params["jec"]["campaign"]
    version = jec_params["jec"]["version"]
    jet_type = jec_params["jet_names"]["AK4"]
    data_mc = "MC"

    systematics = jec_params.get("regrouped_systematics", [])
    if not systematics:
        systematics = jec_params.get("systematics", [])

    for name in systematics:
        key = f"{campaign}_{version}_{data_mc}_{name}_{jet_type}"
        assert key in cset_jec, f"JEC key {key} not found in {jec_file}"

    # 2. Check Btag SF
    btag_params = metadata["era"]["btag_scale_factors"]
    btag_file = btag_params["file"]
    cset_btag = correctionlib.CorrectionSet.from_file(btag_file)
    corr_btag = cset_btag["deepJet_shape"]

    # We expect these systematics to be present in the correction
    # Note: Btag corrections usually handle central, up_xxx, down_xxx as values
    # of the 'systematic' input. Since it's just 'string' type in schema,
    # we can't easily check 'allowed' values, but we can check if it runs
    # for a dummy input.

    syst_names = btag_params["systematics"]
    era_suffix = metadata["era"]["name"]

    # These are the base names from YAML
    for name in syst_names:
        # The code in BJetShapeSF strips the era suffix
        stripped_name = name.removesuffix(era_suffix).rstrip("_")

        for updown in ["up", "down"]:
            full_syst = f"{updown}_{stripped_name}"
            # Try evaluating with dummy values to see if it raises
            try:
                corr_btag.evaluate(full_syst, 0, 0.5, 50.0, 0.5)
            except Exception as e:
                pytest.fail(
                    f"Btag systematic {full_syst} failed evaluation in {btag_file}: {e}"
                )

    # 3. Check JES-correlated systematics in Btag SF
    jes_correlated = btag_params.get("jes_correlated_systematics", [])
    for name in jes_correlated:
        # BJetShapeSF strips "Regrouped_" and adds "jes" prefix
        stripped_name = name.replace("Regrouped_", "")
        for updown in ["up", "down"]:
            full_syst = f"{updown}_jes{stripped_name}"
            try:
                corr_btag.evaluate(full_syst, 0, 0.5, 50.0, 0.5)
            except Exception as e:
                pytest.fail(
                    f"Btag correlated systematic {full_syst} failed evaluation in {btag_file}: {e}"
                )
