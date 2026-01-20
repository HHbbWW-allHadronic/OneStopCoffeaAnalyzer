from analyzer.core.run_builders import (
    toTuples,
    fromTuples,
    buildCombos,
    NoSystematics,
    CompleteSysts,
    SignalOnlySysts,
    MultiRunBuilder,
    RunBuilder,
)
from analyzer.core.param_specs import (
    PipelineParameterSpec,
    ModuleParameterSpec,
    ParameterSpec,
)
import pytest


def testToTuplesFromTuples():
    data = {"a": {"b": 1}, "c": {"d": 2}}
    tuples = toTuples(data)
    assert tuples == {("a", "b"): 1, ("c", "d"): 2}

    reconstructed = fromTuples(tuples)
    assert reconstructed == data


def testBuildCombos():
    p1 = ParameterSpec(default_value="d1", possible_values=["d1", "v1"], tags={"t1"})
    p2 = ParameterSpec(default_value="d2", possible_values=["d2", "v2"], tags={"t1"})

    mod_spec = ModuleParameterSpec(param_specs={"p1": p1, "p2": p2})
    pipeline_spec = PipelineParameterSpec(node_specs={"mod": mod_spec})

    combos = buildCombos(pipeline_spec, "t1")

    names = [c[0] for c in combos]
    assert "mod_p1_v1" in names
    assert "mod_p2_v2" in names

    for name, val in combos:
        if name == "mod_p1_v1":
            assert val["mod"]["p1"] == "v1"
            assert val["mod"]["p2"] == "d2"
        if name == "mod_p2_v2":
            assert val["mod"]["p1"] == "d1"
            assert val["mod"]["p2"] == "v2"


def testNoSystematics():
    builder = NoSystematics()
    spec = PipelineParameterSpec(node_specs={})
    res = builder(spec, {})
    assert len(res) == 1
    assert res[0] == ("central", {})


def testCompleteSysts():
    def make_spec(tag):
        p = ParameterSpec(
            default_value="cen", possible_values=["cen", "up", "down"], tags={tag}
        )
        mod = ModuleParameterSpec(param_specs={"p": p})
        return PipelineParameterSpec(node_specs={"m": mod})

    builder = CompleteSysts()

    spec_w = make_spec("weight_variation")
    res = builder(spec_w, {})

    assert len(res) == 3
    names = [x[0] for x in res]
    assert "central" in names
    assert "m_p_up" in names
    assert "m_p_down" in names

    spec_s = make_spec("shape_variation")
    res = builder(spec_s, {})
    assert len(res) == 3


def testSignalOnlySysts():
    def make_spec(tag):
        p = ParameterSpec(
            default_value="cen", possible_values=["cen", "up"], tags={tag}
        )
        mod = ModuleParameterSpec(param_specs={"p": p})
        return PipelineParameterSpec(node_specs={"m": mod})

    builder = SignalOnlySysts()
    spec = make_spec("weight_variation")

    res = builder(spec, {"dataset_name": "signal_123", "is_signal": True})
    assert len(res) == 2

    res = builder(spec, {"dataset_name": "bkg_123", "is_signal": False})
    assert len(res) == 1
    assert res[0][0] == "central"


def testMultiRunBuilder():
    b1 = NoSystematics()
    b2 = NoSystematics()

    multi = b1 + b2
    assert isinstance(multi, MultiRunBuilder)
    assert len(multi.components) == 2

    res = multi(PipelineParameterSpec(node_specs={}), {})

    assert len(res) == 2
    assert res[0] == ("central", {})
    assert res[1] == ("central", {})


def testComplexSysts():
    def make_param(name, tags):
        return ParameterSpec(
            default_value=f"{name}_def",
            possible_values=[f"{name}_def", f"{name}_up", f"{name}_down"],
            tags=tags,
        )

    m1_specs = {
        "p1": make_param("p1", {"weight_variation"}),
        "p2": make_param("p2", {"shape_variation"}),
        "p3": make_param("p3", set()),
    }

    m2_specs = {
        "q1": make_param("q1", {"weight_variation"}),
        "q2": make_param("q2", {"weight_variation"}),
    }

    pipe_spec = PipelineParameterSpec(
        node_specs={
            "m1": ModuleParameterSpec(param_specs=m1_specs),
            "m2": ModuleParameterSpec(param_specs=m2_specs),
        }
    )

    builder = CompleteSysts()
    runs = builder(pipe_spec, {})

    assert len(runs) == 9

    run_names = [r[0] for r in runs]
    assert "central" in run_names

    def check_var(run_name, mod, param, val):
        matching = [r for r in runs if r[0] == run_name]
        assert len(matching) == 1, (
            f"Expected 1 run named {run_name}, found {len(matching)}"
        )
        _, _config = matching[0]

        config = pipe_spec.getWithValues(_config)

        assert config[mod][param] == val

        if mod == "m1" and param == "p1":
            pass
        else:
            assert config["m1"]["p1"] == "p1_def"

        if mod == "m1" and param == "p2":
            pass
        else:
            assert config["m1"]["p2"] == "p2_def"

        assert config["m1"]["p3"] == "p3_def"

    check_var("m1_p1_p1_up", "m1", "p1", "p1_up")
    check_var("m1_p2_p2_down", "m1", "p2", "p2_down")
    check_var("m2_q1_q1_up", "m2", "q1", "q1_up")
