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


# Concrete implementation for testing Systematics
@define
class SystematicTestModule(AnalyzerModule):
    param_name: str
    output_name: str

    def getParameterSpec(self, metadata):
        from analyzer.core.param_specs import ModuleParameterSpec, ParameterSpec
        return ModuleParameterSpec({
            self.param_name: ParameterSpec(
                default_value="central",
                possible_values=["central", "up", "down"],
                tags={"weight_variation"}
            )
        })

    def inputs(self, metadata):
        return []

    def outputs(self, metadata):
        return [Column(("Weights", self.output_name))]

    def run(self, columns, params):
        val = 1.0
        if params[self.param_name] == "up":
            val = 2.0
        elif params[self.param_name] == "down":
            val = 0.5
        
        columns[Column(("Weights", self.output_name))] = ak.ones_like(columns.events.event_info.run) * val
        return columns, []

@define
class EVENTSFilterModule(AnalyzerModule):
    run_count: int = 0
    def inputs(self, metadata):
        return []
    def outputs(self, metadata):
        return "EVENTS"
    def run(self, columns, params):
        self.run_count += 1
        mask = ak.Array([True, True, True])
        columns.filter(mask)
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

    def testEventSourceCachingWithParams(self):
        @define
        class ParamSourceTest(EventSourceModule):
            run_count: int = 0
            
            def getParameterSpec(self, metadata):
                from analyzer.core.param_specs import ModuleParameterSpec, ParameterSpec
                return ModuleParameterSpec({
                    "src_param":  ParameterSpec(
                        default_value="A",
                        possible_values=["A", "B"]
                    )
                })
                
            def inputs(self, metadata):
                return []
            def outputs(self, metadata):
                return [Column("jets.pt")]
            def run(self, params):
                self.run_count += 1
                val = 100 if params["src_param"] == "A" else 200
                events = ak.Array({"jets": {"pt": [[val]]}})
                return TrackedColumns.fromEvents(
                    events,
                    metadata={"era": {"name": "2018"}},
                    backend=EventBackend.coffea_imm,
                    provenance=0,
                )
                
        module = ParamSourceTest()
        
        res1 = module(ModuleParameterValues({"src_param": "A"}))
        assert module.run_count == 1
        assert ak.all(res1["jets.pt"] == ak.Array([[100]]))
        
        res2 = module(ModuleParameterValues({"src_param": "A"}))
        assert module.run_count == 1
        
        res3 = module(ModuleParameterValues({"src_param": "B"}))
        assert module.run_count == 2
        assert ak.all(res3["jets.pt"] == ak.Array([[200]]))

class TestSystematicsCaching:
    def testMultiParameterCachingProvenance(self, tracked_columns):
        module = SystematicTestModule(param_name="sys1", output_name="weight1")
        
        # Test central
        params_central = ModuleParameterValues({"sys1": "central"})
        res_central, _ = module(tracked_columns, params_central)
        prov_central = res_central._column_provenance[Column("Weights.weight1")]
        
        # Test up
        params_up = ModuleParameterValues({"sys1": "up"})
        res_up, _ = module(tracked_columns, params_up)
        prov_up = res_up._column_provenance[Column("Weights.weight1")]
        
        assert prov_central != prov_up
        assert ak.all(res_central["Weights", "weight1"] == 1.0)
        assert ak.all(res_up["Weights", "weight1"] == 2.0)

    def testEVENTS_OutputCachingDoesNotRevert(self, tracked_columns):
        # Simulate the bug:
        sys_module = SystematicTestModule(param_name="sys1", output_name="weight1")
        filter_module = EVENTSFilterModule()
        
        # Central run
        params_central = ModuleParameterValues({"sys1": "central"})
        c_central, _ = sys_module(tracked_columns, params_central)
        c_central_filt, _ = filter_module(c_central, params_central)
        
        # Up run
        params_up = ModuleParameterValues({"sys1": "up"})
        c_up, _ = sys_module(tracked_columns, params_up)
        c_up_filt, _ = filter_module(c_up, params_up)
        
        # If the bug was present, c_up_filt would have its Weights.weight1 provenance reverted to c_central
        prov_central = c_central_filt._column_provenance[Column("Weights.weight1")]
        prov_up = c_up_filt._column_provenance[Column("Weights.weight1")]
        
        # Verify the changed column correctly maintains isolated provenances
        assert prov_central != prov_up
        assert ak.all(c_up_filt["Weights", "weight1"] == 2.0)
        
        # Verify that an UNCHANGED column (e.g. event_info) has the IDENTICAL provenance 
        # across both cache paths despite passing through a global filter!
        assert c_central_filt._column_provenance[Column("event_info")] == c_up_filt._column_provenance[Column("event_info")]
        
        # Filter module should run twice because the input TrackedColumns provenance changed
        assert filter_module.run_count == 2

    def testMultipleFiltersWithSystematics(self, tracked_columns):
        sys_module = SystematicTestModule(param_name="sys1", output_name="weight1")
        filter1 = EVENTSFilterModule()
        filter2 = EVENTSFilterModule()
        
        params_central = ModuleParameterValues({"sys1": "central"})
        c_central, _ = sys_module(tracked_columns, params_central)
        c_c_f1, _ = filter1(c_central, params_central)
        c_c_f2, _ = filter2(c_c_f1, params_central)
        
        params_up = ModuleParameterValues({"sys1": "up"})
        c_up, _ = sys_module(tracked_columns, params_up)
        c_u_f1, _ = filter1(c_up, params_up)
        c_u_f2, _ = filter2(c_u_f1, params_up)
        
        # Unchanged column provenance must be identical across variations even after TWO independent filters
        assert c_c_f2._column_provenance[Column("event_info")] == c_u_f2._column_provenance[Column("event_info")]
        
        # Changed column provenance must be logically isolated
        assert c_c_f2._column_provenance[Column("Weights.weight1")] != c_u_f2._column_provenance[Column("Weights.weight1")]

    def testSelectOnColumnsCaching(self, tracked_columns):
        from analyzer.modules.common.selection import SelectOnColumns
        
        tc1 = tracked_columns.copy()
        tc1._current_provenance = 10
        tc1[Column("Selection.cut1")] = ak.Array([True, False, True])
        
        sel_module = SelectOnColumns(sel_name="test_sel", selection_names=["cut1"], save_cutflow=False)
        params = ModuleParameterValues()
        
        # First run executes filter and stores cache
        c_out1, _ = sel_module(tc1, params)
        
        # Second identical run must perfectly yield identical TrackedColumns object cached lookup
        c_out2, _ = sel_module(tc1, params)
        assert c_out1 is c_out2
        
        # Change unrelated column -> creates real variation (full EVENTS lookup will alter cache key, missing the initial hit)
        tc2 = tc1.copy()
        tc2._current_provenance = 20
        tc2[Column("jets.pt")] = ak.Array([[99], [99], [99]])
        
        c_out3, _ = sel_module(tc2, params)
        
        # Since underlying struct changed, entire object is newly allocated/evaluated through real filter execution, missing cache perfectly
        assert c_out1 is not c_out3

    def testEVENTS_InputCaching(self, tracked_columns):
        @define
        class EVENTSInputModule(AnalyzerModule):
            run_count: int = 0
            def inputs(self, metadata): return "EVENTS"
            def outputs(self, metadata): return [Column("new_col")]
            def run(self, columns, params):
                self.run_count += 1
                columns[Column("new_col")] = ak.Array([1, 2, 3])
                return columns, []
        
        mod = EVENTSInputModule()
        params = ModuleParameterValues()
        
        c1, _ = mod(tracked_columns, params)
        assert mod.run_count == 1
        
        c2, _ = mod(tracked_columns, params)
        assert mod.run_count == 1
        assert c1 is c2 or c1._column_provenance == c2._column_provenance
        
        tc_mod = tracked_columns.copy()
        tc_mod._current_provenance = 999
        tc_mod[Column("jets.pt")] = ak.Array([[1], [1], [1]])
        
        c3, _ = mod(tc_mod, params)
        assert mod.run_count == 2

