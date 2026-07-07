import pytest
from analyzer.core.param_specs import ParameterSpec, ModuleParameterSpec, getWithValues
from analyzer.core.run_builders import (
    buildCombos,
    _buildDrivenMap,
    CompleteSysts,
    WeightsOnly,
)


def _makeJesBtagSpec():
    """Build a realistic combined spec with JES shape + btag SF weight params.

    Returns the full ModuleParameterSpec dict.
    """

    def jes_to_btag(jes_val):
        if jes_val == "central":
            return None
        return jes_val

    spec = ModuleParameterSpec(
        {
            "jes-variation": ParameterSpec(
                default_value="central",
                possible_values=[
                    "central",
                    "up_jesAbsolute",
                    "down_jesAbsolute",
                    "up_jesFlavorQCD",
                    "down_jesFlavorQCD",
                ],
                tags={"shape_variation", "jes"},
            ),
            "bjetshapesf-variation": ParameterSpec(
                default_value="central",
                possible_values=[
                    "central",
                    "up_hf",
                    "down_hf",
                    "up_lf",
                    "down_lf",
                    # JES-correlated values
                    "up_jesAbsolute",
                    "down_jesAbsolute",
                    "up_jesFlavorQCD",
                    "down_jesFlavorQCD",
                ],
                tags={"weight_variation"},
                driven_by={"jes-variation": jes_to_btag},
            ),
        }
    )
    return spec


# ---------------------------------------------------------------------------
# ParameterSpec.getIndependentValues tests
# ---------------------------------------------------------------------------


class TestParameterSpecIndependentValues:
    def testNoDriverReturnsAllValues(self):
        """A param with no driven_by returns all possible_values."""
        spec = ParameterSpec(
            default_value="central",
            possible_values=["central", "up", "down"],
        )
        assert spec.getIndependentValues({}) == {"central", "up", "down"}

    def testDrivenValuesExcluded(self):
        """Values reachable via driven_by mapping are excluded."""
        full_spec = _makeJesBtagSpec()
        btag_spec = full_spec["bjetshapesf-variation"]
        independent = btag_spec.getIndependentValues(full_spec)

        # JES-correlated values should be excluded
        assert "up_jesAbsolute" not in independent
        assert "down_jesAbsolute" not in independent
        assert "up_jesFlavorQCD" not in independent
        assert "down_jesFlavorQCD" not in independent

        # Independent btag values remain
        assert "central" in independent
        assert "up_hf" in independent
        assert "down_hf" in independent
        assert "up_lf" in independent
        assert "down_lf" in independent

    def testDriverNotInSpecReturnsAllValues(self):
        """If driven_by references a param not in the full spec, all values
        are returned (no driven values to subtract)."""

        def mapping(v):
            return v if v != "central" else None

        spec = ParameterSpec(
            default_value="central",
            possible_values=["central", "up", "down", "up_jesX"],
            driven_by={"nonexistent-param": mapping},
        )
        independent = spec.getIndependentValues({})
        assert independent == {"central", "up", "down", "up_jesX"}

    def testEmptyPossibleValues(self):
        spec = ParameterSpec(possible_values=None)
        assert spec.getIndependentValues({}) == set()

    def testMappingReturningNoneKeepsValues(self):
        """If the mapping returns None for all driver values, nothing is excluded."""

        def always_none(v):
            return None

        full_spec = {
            "driver": ParameterSpec(
                default_value="central",
                possible_values=["central", "up"],
            ),
            "driven": ParameterSpec(
                default_value="X",
                possible_values=["X", "Y", "Z"],
                driven_by={"driver": always_none},
            ),
        }
        assert full_spec["driven"].getIndependentValues(full_spec) == {"X", "Y", "Z"}


class TestBuildDrivenMap:
    def testEmptySpec(self):
        assert _buildDrivenMap({}) == {}

    def testNoDrivenBy(self):
        spec = {
            "a": ParameterSpec(possible_values=["x", "y"]),
            "b": ParameterSpec(possible_values=["1", "2"]),
        }
        assert _buildDrivenMap(spec) == {}

    def testSingleCorrelation(self):
        def identity(v):
            return v

        spec = {
            "driver": ParameterSpec(possible_values=["central", "up"]),
            "driven": ParameterSpec(
                possible_values=["central", "up"],
                driven_by={"driver": identity},
            ),
        }
        driven_map = _buildDrivenMap(spec)
        assert "driver" in driven_map
        assert len(driven_map["driver"]) == 1
        assert driven_map["driver"][0] == ("driven", identity)

    def testMultipleDrivers(self):
        """A single param can be driven by multiple other params."""

        def identity(v):
            return v

        def prefixed(v):
            return f"mapped_{v}"

        spec = {
            "d1": ParameterSpec(possible_values=["a"]),
            "d2": ParameterSpec(possible_values=["b"]),
            "target": ParameterSpec(
                possible_values=["a", "b", "mapped_b"],
                driven_by={"d1": identity, "d2": prefixed},
            ),
        }
        driven_map = _buildDrivenMap(spec)
        assert "d1" in driven_map
        assert "d2" in driven_map


class TestBuildCombos:
    def testBasicWeightCombosUnchanged(self):
        """When no correlations exist, buildCombos behaves exactly as before."""
        spec = {
            "w1": ParameterSpec(
                default_value="central",
                possible_values=["central", "up", "down"],
                tags={"weight_variation"},
            ),
            "w2": ParameterSpec(
                default_value="nom",
                possible_values=["nom", "high", "low"],
                tags={"weight_variation"},
            ),
        }
        combos = buildCombos(spec, "weight_variation")
        names = [c[0] for c in combos]

        assert "w1_up" in names
        assert "w1_down" in names
        assert "w2_high" in names
        assert "w2_low" in names
        assert len(combos) == 4

    def testCorrelatedJesBtagCombos(self):
        """When JES is varied, btag SF is automatically set to the correlated value."""
        spec = _makeJesBtagSpec()

        shapes = buildCombos(spec, "shape_variation")
        shape_names = [c[0] for c in shapes]
        shape_dicts = {c[0]: c[1] for c in shapes}

        # JES variations should exist
        assert "jes-variation_up_jesAbsolute" in shape_names
        assert "jes-variation_down_jesAbsolute" in shape_names

        # When JES is varied, btag SF should be correlated
        jes_up_abs = shape_dicts["jes-variation_up_jesAbsolute"]
        assert jes_up_abs["jes-variation"] == "up_jesAbsolute"
        assert jes_up_abs["bjetshapesf-variation"] == "up_jesAbsolute"

        jes_down_qcd = shape_dicts["jes-variation_down_jesFlavorQCD"]
        assert jes_down_qcd["jes-variation"] == "down_jesFlavorQCD"
        assert jes_down_qcd["bjetshapesf-variation"] == "down_jesFlavorQCD"

    def testDrivenValuesNotVariedIndependently(self):
        """JES-correlated btag SF values must NOT appear as independent weight variations."""
        spec = _makeJesBtagSpec()

        weights = buildCombos(spec, "weight_variation")
        weight_names = [c[0] for c in weights]

        # Independent btag SF values should be present
        assert "bjetshapesf-variation_up_hf" in weight_names
        assert "bjetshapesf-variation_down_lf" in weight_names

        # JES-correlated btag SF values should NOT be independently varied
        for name in weight_names:
            assert "jesAbsolute" not in name, (
                f"Driven value appeared independently: {name}"
            )
            assert "jesFlavorQCD" not in name, (
                f"Driven value appeared independently: {name}"
            )

    def testNonCorrelatedParamUnaffected(self):
        """A shape param without driven_by targets keeps btag SF at default."""

        def jes_to_btag(v):
            return v if v != "central" else None

        spec = {
            "jes-variation": ParameterSpec(
                default_value="central",
                possible_values=["central", "up_jesX"],
                tags={"shape_variation"},
            ),
            "jer-variation": ParameterSpec(
                default_value="nom",
                possible_values=["nom", "up_JER"],
                tags={"shape_variation"},
            ),
            "bjetshapesf-variation": ParameterSpec(
                default_value="central",
                possible_values=["central", "up_hf", "up_jesX"],
                tags={"weight_variation"},
                driven_by={"jes-variation": jes_to_btag},
            ),
        }
        shapes = buildCombos(spec, "shape_variation")
        shape_dicts = {c[0]: c[1] for c in shapes}

        # JER variation should not set btag SF
        jer_combo = shape_dicts["jer-variation_up_JER"]
        assert "bjetshapesf-variation" not in jer_combo

        # JES variation should set btag SF
        jes_combo = shape_dicts["jes-variation_up_jesX"]
        assert jes_combo["bjetshapesf-variation"] == "up_jesX"


class TestRunBuildersWithCorrelations:
    def testCompleteSystsIncludesCorrelations(self):
        """CompleteSysts should produce correlated btag SF in JES combos."""
        spec = _makeJesBtagSpec()
        builder = CompleteSysts()
        combos = builder(spec, metadata={})

        combo_dict = {c[0]: c[1] for c in combos}

        # Central exists
        assert "central" in combo_dict

        # JES combo has correlated btag
        assert "jes-variation_up_jesAbsolute" in combo_dict
        assert (
            combo_dict["jes-variation_up_jesAbsolute"]["bjetshapesf-variation"]
            == "up_jesAbsolute"
        )

        # Independent weight combos exist
        assert "bjetshapesf-variation_up_hf" in combo_dict

        # Independent weight combos do NOT include JES-correlated btag values
        for name in combo_dict:
            if name.startswith("bjetshapesf-variation_"):
                assert "jesAbsolute" not in name
                assert "jesFlavorQCD" not in name

    def testWeightsOnlyExcludesDrivenValues(self):
        """WeightsOnly should only produce independent btag SF variations."""
        spec = _makeJesBtagSpec()
        builder = WeightsOnly()
        combos = builder(spec, metadata={})
        combo_names = [c[0] for c in combos]

        assert "central" in combo_names
        assert "bjetshapesf-variation_up_hf" in combo_names

        for name in combo_names:
            if name == "central":
                continue
            assert "jesAbsolute" not in name
            assert "jesFlavorQCD" not in name


class TestGetWithValuesCorrelated:
    def testCorrelatedValueAccepted(self):
        """getWithValues should accept correlated values that are in possible_values."""
        spec = _makeJesBtagSpec()
        values = {
            "jes-variation": "up_jesAbsolute",
            "bjetshapesf-variation": "up_jesAbsolute",
        }
        result = getWithValues(spec, values)
        assert result["jes-variation"] == "up_jesAbsolute"
        assert result["bjetshapesf-variation"] == "up_jesAbsolute"

    def testInvalidValueRejected(self):
        """getWithValues should reject values not in possible_values."""
        spec = _makeJesBtagSpec()
        values = {
            "jes-variation": "central",
            "bjetshapesf-variation": "up_jesNONEXISTENT",
        }
        with pytest.raises(RuntimeError, match="not in the list of possible values"):
            getWithValues(spec, values)

    def testDefaultsApplied(self):
        """When no value given, defaults are used."""
        spec = _makeJesBtagSpec()
        result = getWithValues(spec, {})
        assert result["jes-variation"] == "central"
        assert result["bjetshapesf-variation"] == "central"
