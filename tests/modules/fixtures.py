import awkward as ak
from pathlib import Path
from analyzer.core.columns import TrackedColumns, Column, EventBackend
from typing import Any
import pytest


def createMockMetadata(
    era_name: str = "2018",
    sample_type: str = "MC",
    dataset_name: str = "test_dataset",
    sample_name: str = "test_sample",
    **extra_fields,
) -> dict[str, Any]:
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    era_file = (
        Path(__file__).parent.parent.parent / f"analyzer_resources/eras/{era_name}.yaml"
    )

    if not era_file.exists():
        raise FileNotFoundError(f"Era file not found: {era_file}")

    with open(era_file, "r") as f:
        era_data = yaml.safe_load(f)

    if isinstance(era_data, list):
        era_config = era_data[0]
    else:
        era_config = era_data

    metadata = {
        "era": era_config,
        "sample_type": sample_type,
        "dataset_name": dataset_name,
        "sample_name": sample_name,
        "chunk": {
            "file_path": "/path/to/file.root",
            "event_start": 0,
            "event_stop": 100,
        },
    }
    metadata.update(extra_fields)
    return metadata


@pytest.mark.filterwarnings("ignore::RuntimeWarning:coffea.*")
def createMockEvents(
    n_events: int = 10,
    start_event: int = 0,
    test_file: str = "tests/test_data/nano_dy.root",
) -> ak.Array:
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    file_path = Path(__file__).parent.parent.parent / test_file

    if not file_path.exists():
        file_path = Path(test_file)

    events = NanoEventsFactory.from_root(
        {str(file_path): "Events"},
        schemaclass=NanoAODSchema,
        entry_start=start_event,
        entry_stop=start_event + n_events,
    ).events()

    return events


def createMockTrackedColumns(
    events: ak.Array | None = None,
    metadata: dict | None = None,
    backend: EventBackend = EventBackend.coffea_imm,
) -> TrackedColumns:
    if events is None:
        events = createMockEvents()

    if metadata is None:
        metadata = createMockMetadata()

    columns = TrackedColumns.fromEvents(events, metadata, backend, provenance=0)
    return columns


def assertColumnsEqual(
    col1: ak.Array, col2: ak.Array, rtol: float = 1e-5, atol: float = 1e-8
):
    if col1.ndim != col2.ndim:
        raise AssertionError(
            f"Arrays have different dimensions: {col1.ndim} vs {col2.ndim}"
        )
    flat1 = ak.flatten(col1) if col1.ndim > 1 else col1
    flat2 = ak.flatten(col2) if col2.ndim > 1 else col2

    if len(flat1) != len(flat2):
        raise AssertionError(
            f"Arrays have different lengths: {len(flat1)} vs {len(flat2)}"
        )
    if not ak.all(ak.isclose(flat1, flat2, rtol=rtol, atol=atol)):
        raise AssertionError("Arrays are not equal within tolerance")


def assertColumnExists(columns: TrackedColumns, column: Column | str):
    if isinstance(column, str):
        column = Column(column)

    try:
        _ = columns[column]
    except (KeyError, AttributeError) as e:
        raise AssertionError(f"Column {column} does not exist in TrackedColumns") from e


def assertColumnShape(
    columns: TrackedColumns, column: Column | str, expected_ndim: int
):
    if isinstance(column, str):
        column = Column(column)

    data = columns[column]
    if data.ndim != expected_ndim:
        raise AssertionError(
            f"Column {column} has {data.ndim} dimensions, expected {expected_ndim}"
        )
