import pytest
import awkward as ak
from attrs import define, field
from analyzer.core.analysis_modules import AnalyzerModule, EventSourceModule
from analyzer.core.columns import TrackedColumns, EventBackend, Column
from analyzer.core.param_specs import ModuleParameterValues


# Mock Data
@pytest.fixture
def sample_events():
    return ak.Array(
        {
            "event_info": {
                "run": [1, 1, 1],
                "lumiblock": [1, 1, 1],
                "event": [1, 2, 3],
            },
            "jets": {
                "pt": [[10, 15], [20], [30, 35, 40]],
                "eta": [[0.1, -0.4], [1.2], [-0.5, 0.5, 0.0]],
                "mass": [[1, 2], [2], [3, 4, 5]],
            },
        }
    )


@pytest.fixture
def tracked_columns(sample_events):
    return TrackedColumns.fromEvents(
        sample_events,
        metadata={"era": {"name": "2018"}, "sample_type": "data"},
        backend=EventBackend.coffea_imm,
        provenance=0,
    )


# Concrete implementation for testing AnalyzerModule
@define
class AnalyzerTest(AnalyzerModule):
    input_cols: list[str] = field(factory=list)
    output_cols: list[str] = field(factory=list)
    run_count: int = 0

    def inputs(self, metadata):
        return [Column(x) for x in self.input_cols]

    def outputs(self, metadata):
        return [Column(x) for x in self.output_cols]

    def run(self, columns, params):
        self.run_count += 1
        if "out" in self.output_cols:
            if "jets.pt" in self.input_cols:
                columns["out"] = columns["jets.pt"] * 2
            else:
                columns["out"] = ak.Array([[0], [0], [0]])
        return columns, []


class TestAnalyzerModule:
    def testRunCaching(self, tracked_columns):
        module = AnalyzerTest(input_cols=["jets.pt"], output_cols=["out"])
        params = ModuleParameterValues()

        res1, _ = module(tracked_columns, params)
        assert module.run_count == 1
        assert ak.all(res1["out"] == tracked_columns["jets.pt"] * 2)
        assert Column("out") in res1._column_provenance

        res2, _ = module(tracked_columns, params)
        assert module.run_count == 1
        assert ak.all(res2["out"] == res1["out"])

    def testKeyGenerationSensitivity(self, tracked_columns):
        module = AnalyzerTest(input_cols=["jets.pt"], output_cols=["out"])
        params1 = ModuleParameterValues()
        params2 = ModuleParameterValues()

        key1 = module.getKey(tracked_columns, params1)
        key2 = module.getKey(tracked_columns, params2)
        assert key1 == key2
        tracked_columns._current_provenance = 1
        tracked_columns["jets.pt"] = ak.Array([[1], [1], [1]])
        key3 = module.getKey(tracked_columns, params1)
        assert key1 != key3


@define
class SourceTest(EventSourceModule):
    run_count: int = 0

    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return [Column("jets.pt")]

    def run(self, params):
        self.run_count += 1
        events = ak.Array({"jets": {"pt": [[100], [200], [300]]}})
        return TrackedColumns.fromEvents(
            events,
            metadata={"era": {"name": "2018"}},
            backend=EventBackend.coffea_imm,
            provenance=0,
        )


class TestEventSourceModule:
    def testRunCaching(self):
        module = SourceTest()
        params = ModuleParameterValues()

        res1 = module(params)
        assert module.run_count == 1
        assert ak.all(res1["jets.pt"] == ak.Array([[100], [200], [300]]))
        res2 = module(params)
        assert module.run_count == 1
        assert res1 is res2
