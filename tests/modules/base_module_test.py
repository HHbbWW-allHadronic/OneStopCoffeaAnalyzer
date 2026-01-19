import pytest
from typing import Any
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    EventSourceModule,
    PureResultModule,
    ModuleParameterValues,
)
from analyzer.core.columns import TrackedColumns, Column
from .fixtures import (
    createMockMetadata,
    createMockEvents,
    createMockTrackedColumns,
    assertColumnsEqual,
    assertColumnExists,
    assertColumnShape,
)


class BaseModuleTest:
    @pytest.fixture
    def mockMetadata(self) -> dict[str, Any]:
        return createMockMetadata()

    @pytest.fixture
    def mockEvents(self) -> Any:
        return createMockEvents(n_events=50)

    @pytest.fixture
    def mockColumns(self, mockEvents, mockMetadata) -> TrackedColumns:
        return createMockTrackedColumns(mockEvents, mockMetadata)

    def runModule(
        self,
        module: AnalyzerModule,
        columns: TrackedColumns,
        params: dict | None = None,
    ) -> tuple[TrackedColumns, list]:
        if params is None:
            params = {}

        param_values = ModuleParameterValues(params)
        return module(columns, param_values)

    def runEventSourceModule(
        self, module: EventSourceModule, params: dict | None = None
    ) -> TrackedColumns:
        if params is None:
            params = {}

        param_values = ModuleParameterValues(params)
        return module(param_values)

    def runPureResultModule(
        self,
        module: PureResultModule,
        column_sets: list[tuple[Any, TrackedColumns]],
        params: dict | None = None,
    ) -> list:
        if params is None:
            params = {}

        param_values = ModuleParameterValues(params)
        return module(column_sets, param_values)

    def assertInputsCorrect(self, module: AnalyzerModule, metadata: dict):
        inputs = module.inputs(metadata)
        assert inputs is not None, "inputs() returned None"

        if inputs != "EVENTS":
            assert isinstance(inputs, (list, tuple)), (
                "inputs() must return list, tuple, or 'EVENTS'"
            )
            for inp in inputs:
                assert isinstance(inp, Column), f"Input {inp} is not a Column object"

    def assertOutputsCorrect(
        self, module: AnalyzerModule | PureResultModule, metadata: dict
    ):
        outputs = module.outputs(metadata)
        assert outputs is not None, "outputs() returned None"

        if outputs != "EVENTS":
            assert isinstance(outputs, (list, tuple)), (
                "outputs() must return list, tuple, or 'EVENTS'"
            )
            for out in outputs:
                assert isinstance(out, Column), f"Output {out} is not a Column object"

    def assertParameterSpecCorrect(self, module: AnalyzerModule, metadata: dict):
        spec = module.getParameterSpec(metadata)
        assert spec is not None, "getParameterSpec() returned None"
        for param_name, param_spec in spec.items():
            if hasattr(param_spec, "default_value") and hasattr(
                param_spec, "possible_values"
            ):
                if param_spec.possible_values is not None:
                    assert param_spec.default_value in param_spec.possible_values, (
                        f"Default value {param_spec.default_value} not in possible values for {param_name}"
                    )

    def assertModuleRunsWithoutError(
        self,
        module: AnalyzerModule,
        columns: TrackedColumns,
        params: dict | None = None,
    ):
        try:
            self.runModule(module, columns, params)
        except Exception as e:
            pytest.fail(f"Module raised an exception: {e}")

    def assertModuleProducesOutputs(
        self,
        module: AnalyzerModule,
        columns: TrackedColumns,
        expected_outputs: list[Column | str],
        params: dict | None = None,
    ):
        output_columns, _ = self.runModule(module, columns, params)

        for expected_output in expected_outputs:
            assertColumnExists(output_columns, expected_output)

    def assertCachingWorks(
        self,
        module: AnalyzerModule,
        columns: TrackedColumns,
        params: dict | None = None,
    ):
        module.clearCache()
        result1, outputs1 = self.runModule(module, columns, params)
        result2, outputs2 = self.runModule(module, columns, params)
        assert len(outputs1) == len(outputs2), (
            "Different number of outputs from cached run"
        )

    def assertModuleRespectsMetadataCondition(
        self,
        module: AnalyzerModule,
        columns_should_run: TrackedColumns,
        columns_should_not_run: TrackedColumns,
        params: dict | None = None,
    ):
        if module.should_run is None:
            pytest.skip("Module does not have should_run condition")
        result_run, outputs_run = self.runModule(module, columns_should_run, params)
        result_skip, outputs_skip = self.runModule(
            module, columns_should_not_run, params
        )
        assert len(outputs_skip) == 0, (
            "Module produced outputs when it should have been skipped"
        )
