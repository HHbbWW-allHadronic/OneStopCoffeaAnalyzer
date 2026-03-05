import pytest
import awkward as ak
from analyzer.core.analyzer import Analyzer
from analyzer.core.analysis_modules import AnalyzerModule, EventSourceModule
from analyzer.core.param_specs import ModuleParameterSpec, ParameterSpec
from analyzer.core.columns import Column, TrackedColumns, EventBackend
from attrs import define

@define
class HardSource(EventSourceModule):
    run_count: int = 0
    def inputs(self, metadata): return []
    def outputs(self, metadata): return [Column("jets.pt"), Column("jets.eta"), Column("jets.phi")]
    def run(self, params):
        self.run_count += 1
        arr = ak.Array({
            "event_info": {"run": [1, 1, 1]},
            "jets": {
                "pt": [[10, 20], [30, 40], [50, 60]],
                "eta": [[0.5, 1.5], [2.5, 3.5], [-1.0, 0.0]],
                "phi": [[3.1, 0.0], [1.5, -1.5], [0.5, -0.5]]
            }
        })
        tc = TrackedColumns.fromEvents(arr, {}, EventBackend.coffea_imm, 0)
        return tc

@define
class JetPtSyst(AnalyzerModule):
    run_count: int = 0
    def getParameterSpec(self, metadata):
        return ModuleParameterSpec({
            "jes": ParameterSpec("central", ["central", "up", "down"])
        })
    def inputs(self, metadata): return [Column("jets")]
    def outputs(self, metadata): return [Column("jets.pt_sys")]
    def run(self, columns, params):
        self.run_count += 1
        jes = params["jes"]
        mult = 1.0 if jes == "central" else (1.1 if jes == "up" else 0.9)
        columns["jets.pt_sys"] = columns["jets.pt"] * mult
        return columns, []

@define
class SmartSelection(AnalyzerModule):
    run_count: int = 0
    def inputs(self, metadata): return [Column("jets")]
    def outputs(self, metadata): return "EVENTS"
    def run(self, columns, params):
        self.run_count += 1
        # Filter events where at least one jet has pt_sys > 25
        mask = ak.num(columns["jets.pt_sys"][columns["jets.pt_sys"] > 25]) > 0
        columns.filter(mask)
        return columns, []

@define
class WeightSyst(AnalyzerModule):
    run_count: int = 0
    def getParameterSpec(self, metadata):
        return ModuleParameterSpec({
            "weight_sys": ParameterSpec("central", ["central", "up", "down"])
        })
    def inputs(self, metadata): return [Column("jets")]
    def outputs(self, metadata): return [Column("Weight")]
    def run(self, columns, params):
        self.run_count += 1
        sys = params["weight_sys"]
        val = 1.0 if sys == "central" else (1.2 if sys == "up" else 0.8)
        columns["Weight"] = ak.Array([val] * len(columns["jets.eta"]))
        return columns, []

@define
class PhiEvaluator(AnalyzerModule):
    run_count: int = 0
    def inputs(self, metadata): return [Column("jets")]
    def outputs(self, metadata): return [Column("PhiResult")]
    def run(self, columns, params):
        self.run_count += 1
        columns["PhiResult"] = columns["jets.phi"] * 2
        return columns, []

@define
class FinalEvaluator(AnalyzerModule):
    run_count: int = 0
    def inputs(self, metadata): return [Column("jets"), Column("Weight")]
    def outputs(self, metadata): return [Column("Result")]
    def run(self, columns, params):
        self.run_count += 1
        columns["Result"] = ak.sum(columns["jets.pt_sys"], axis=1) * columns["Weight"]
        return columns, []

@define
class DummyHist(AnalyzerModule):
    run_count: int = 0
    def inputs(self, metadata): return []
    def outputs(self, metadata): return []
    def run(self, columns, params):
        self.run_count += 1
        from analyzer.core.analysis_modules import ModuleAddition, PureResultModule
        from analyzer.core.results import ResultGroup
        
        class MockHistBuilder(PureResultModule):
            def inputs(self, metadata): return []
            def outputs(self, metadata): return []
            def run(self, column_sets, params):
                return [ResultGroup("hists")]
                
        class CrossBuilder:
            def __call__(self, spec, metadata):
                import itertools
                jes_vals = spec["jes"].possible_values
                wsys_vals = spec["weight_sys"].possible_values
                ret = []
                for j, w in itertools.product(jes_vals, wsys_vals):
                    ret.append((f"{j}_{w}", {"jes": j, "weight_sys": w}))
                return ret
                
        return columns, [ModuleAddition(MockHistBuilder(), run_builder=CrossBuilder())]

class TestAnalyzerHard:
    def testComplexPipelineCaching(self):
        source = HardSource()
        jsys = JetPtSyst()
        sel = SmartSelection()
        wsys = WeightSyst()
        phi_eval = PhiEvaluator()
        evaluator = FinalEvaluator()
        
        dummy_hist = DummyHist()
        
        analyzer = Analyzer()
        analyzer.base_pipelines["test"] = [source, jsys, sel, wsys, phi_eval, evaluator, dummy_hist]
        
        from analyzer.core.event_collection import FileChunk
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".root") as tmp:
            chunk = FileChunk(tmp.name, 0, 100, "Events", 100)
            metadata = {
                "dataset_name": "test",
                "sample_name": "test",
                "sample_type": "MC",
                "era": {"name": "2018"},
                "is_signal": False
            }
            
            # This executes the pipeline through osca.py pattern
            # The DummyHist strictly triggers `Analyzer` to branch and recurse permutations (3 jes * 3 weight = 9 variations)
            _ = analyzer.run(chunk, metadata)
        
        # Source yields identical base state, hitting cache automatically after initial pass
        assert source.run_count == 1
        
        # JetPtSyst strictly varies on 'jes', ignoring 'weight_sys'. (3 variations evaluated)
        assert jsys.run_count == 3
        
        # SmartSelection applies EVENT mask strictly off 'jets.pt_sys'. Since 'jets.pt_sys' uniquely hashes 3 ways,
        # it intrinsically yields exactly 3 separated filter masks natively!
        assert sel.run_count == 3
        
        # WeightSyst strictly evaluates off 'jets.eta' and 'weight_sys'.
        # Since 'jets.eta' takes ON the provenance of the 3 masks created by 'sel',
        # Verify visually
        assert wsys.run_count == 9
        
        # CRITICAL VALIDATION: 'jets.phi' never sees WeightSyst
        # Our `TrackedColumns.filter()` independence fix perfectly ensures 'phi' receives JUST the 3 mask provenances.
        # It misses cache exactly 3 times, not blindly 9! 
        assert phi_eval.run_count == 3
        
        # Evaluator assesses the two coupled cross-variations combinations, tracking the 9 distinct paths
        assert evaluator.run_count == 9
