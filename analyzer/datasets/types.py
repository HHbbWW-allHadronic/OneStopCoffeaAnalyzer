from dataclasses import dataclass
from typing import Any, Hashable


from analyzer.configuration import CONFIG


if CONFIG.PRETTY_MODE:
    pass

@dataclass
class UprootFileSpec:
    object_path: str
    steps: list[list[int]] | list[int] | None


@dataclass
class CoffeaFileSpec(UprootFileSpec):
    steps: list[list[int]]
    num_entries: int
    uuid: str


@dataclass
class CoffeaFileSpecOptional(CoffeaFileSpec):
    steps: list[list[int]] | None
    num_entries: int | None
    uuid: str | None


@dataclass
class DatasetSpec:
    files: dict[str, CoffeaFileSpec]
    metadata: dict[Hashable, Any] | None
    form: str | None


@dataclass
class DatasetSpecOptional(DatasetSpec):
    files: (
        dict[str, str] | list[str] | dict[str, UprootFileSpec | CoffeaFileSpecOptional]
    )


FilesetSpecOptional = dict[str, DatasetSpecOptional]
FilesetSpec = dict[str, DatasetSpec]
