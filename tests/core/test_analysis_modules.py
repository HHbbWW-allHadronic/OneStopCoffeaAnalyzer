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
        for out in self.output_cols:
            if len(self.input_cols) > 0:
                columns[out] = columns[self.input_cols[0]] * 2
            else:
                columns[out] = ak.Array([[0], [0], [0]])
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

    def testLazyColumnModuleOutputs(self, tracked_columns):
        module = AnalyzerTest(input_cols=["jets.pt"], output_cols=["out"])
        params = ModuleParameterValues()
        
        res, _ = module(tracked_columns, params)
        
        # Ensure that the module output generated a lazy hook instead of flushing the dataset prematurely
        assert Column("out") in res._lazy_columns
        assert "out" not in res._events.fields
        
        # The key correctly incorporates the proxy update
        key1 = module.getKey(res, params)
        res._current_provenance = 2
        res["jets.pt"] = ak.Array([[9], [9], [9]])
        key2 = module.getKey(res, params)
        
        assert key1 != key2

    def testLazyColumnPipelineBranching(self, tracked_columns):
        # Simulate a root module that puts multiple columns in the lazy dictionary
        module1 = AnalyzerTest(input_cols=["jets.pt"], output_cols=["out_m1"])
        params = ModuleParameterValues()
        
        tc_main, _ = module1(tracked_columns, params)
        assert Column("out_m1") in tc_main._lazy_columns
        
        # Branch the pipeline logic
        tc_b1 = tc_main.copy()
        tc_b2 = tc_main.copy()
        
        # Branch 1 executes Module 2 and applies an event filter
        module2 = AnalyzerTest(input_cols=["out_m1"], output_cols=["out_b1"])
        tc_b1, _ = module2(tc_b1, params)
        
        # Branch 2 executes Module 3 (no filter)
        module3 = AnalyzerTest(input_cols=["jets.pt"], output_cols=["out_b2"])
        tc_b2, _ = module3(tc_b2, params)
        
        mask = ak.Array([True, False, True]) # length 3 events inside tracked_columns -> length 2 
        tc_b1.filter(mask)
        
        # Validation 1: Length Isolation (Event Counts)
        assert len(tc_main._events) == 3
        assert len(tc_main["out_m1"]) == 3
        
        assert len(tc_b1._events) == 2
        assert len(tc_b1["out_m1"]) == 2
        assert len(tc_b1["out_b1"]) == 2
        
        assert len(tc_b2._events) == 3
        assert len(tc_b2["out_m1"]) == 3
        assert len(tc_b2["out_b2"]) == 3
        
        # Validation 2: Column Isolation
        assert "out_b2" not in tc_b1.fields
        assert "out_b1" not in tc_b2.fields
        assert "out_b1" not in tc_main.fields
        assert "out_b2" not in tc_main.fields
        
        # Validation 3: Base Lazy-Proxy Instance Independence
        assert Column("out_b1") in tc_b1._lazy_columns
        assert Column("out_b2") in tc_b2._lazy_columns
        assert tc_main._lazy_columns is not tc_b1._lazy_columns
        assert tc_main._lazy_columns is not tc_b2._lazy_columns
        
        # Validation 4: Underlying Object Isolation (ensure proxies didn't bleed during the memory update)
        assert len(tc_b1._lazy_columns[Column("out_m1")]) == 2
        assert len(tc_b2._lazy_columns[Column("out_m1")]) == 3
        assert len(tc_main._lazy_columns[Column("out_m1")]) == 3


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
