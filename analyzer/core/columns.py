from __future__ import annotations

from attrs import define, field


import contextlib
import functools as ft
import copy
import awkward as ak
from typing import Any
import awkward as ak
import logging
import enum
from analyzer.utils.structure_tools import freeze

logger = logging.getLogger("analyzer.core")


def coerceFields(data):
    if isinstance(data, str):
        return tuple(data.split("."))
    elif isinstance(data, Column):
        return data.fields
    else:
        return data


EVENTS = object()


class EventBackend(str, enum.Enum):
    coffea_virtual = "coffea_virtual"
    coffea_dask = "coffea_dask"
    coffea_imm = "coffea_eager"
    rdf = "rdf"


@define(frozen=True)
class Column:
    fields: tuple[str, ...] = field(converter=coerceFields)

    def contains(self, other):
        if len(self) > len(other):
            return False
        return other[: len(self)] == self

    def extract(self, events):
        return ft.reduce(lambda x, y: x[y], self.fields, events)

    def __iter__(self):
        return iter(self.fields)

    def __eq__(self, other):
        return self.fields == other.fields

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, key):
        return Column(self.fields.__getitem__(key))

    def __add__(self, other):
        return Column(self.fields + Column(other).fields)

    def __radd__(self, other):
        return Column(Column(other).fields + self.fields)

    def __hash__(self):
        return hash(self.fields)

    def __str__(self):
        return ".".join(self.fields)

    def __repr__(self):
        return ".".join(self.fields)

    @classmethod
    def _structure(cls, data: str, conv):
        if isinstance(data, str):
            return Column(data)


def setColumn(events, column, value):
    column = Column(column)
    if len(column) == 1:
        return ak.with_field(events, value, column.fields)
    head, *rest = tuple(column)
    if head not in events.fields:
        for c in reversed(list(rest)):
            value = ak.zip({c: value})
        return ak.with_field(events, value, head)
    else:
        return ak.with_field(events, setColumn(events[head], Column(rest), value), head)


@define
class ColumnCollection:
    columns: set[Column] = field(converter=lambda z: set(Column(x) for x in z))

    def __iter__(self):
        return iter(self.columns)

    def contains(self, other: Column):
        return any(x.contains(other) for x in self.columns)

    def intersect(self, other: ColumnCollection):
        ret = {
            x
            for x in self.columns
            if any((x.contains(o) or o.contains(x)) for o in other)
        }
        return ret


def getAllColumns(events, cur_col=None, cur_depth=0, max_depth=None) -> set[Column]:
    if fields := getattr(events, "fields"):
        ret = set()
        for f in fields:
            if cur_col is not None:
                n = cur_col + f
            else:
                n = Column(f)
            if cur_depth == max_depth:
                ret.add(n)
            else:
                ret |= getAllColumns(
                    events[f], n, max_depth=max_depth, cur_depth=cur_depth + 1
                )

        if cur_col is not None:
            ret.add(cur_col)
        return ret
    else:
        return {cur_col}


@define
class TrackedColumns:
    INTERNAL_USE_COL: ClassVar[Column] = Column("INTERNAL_USE")
    _events: Any
    _column_provenance: dict[Column, int]
    backend: EventBackend
    _current_provenance: int
    _allowed_inputs: ColumnCollection | None = None
    _allowed_outputs: ColumnCollection | None = None
    _allow_filter: bool = True
    metadata: Any | None = None
    pipeline_data: dict[str, Any] = field(factory=dict)

    @property
    def fields(self):
        return self._events.fields

    def updatedColumns(self, old, limit=None):
        cols_to_consider = {
            x: y
            for x, y in self._column_provenance.items()
            if limit is None or limit.contains(x)
        }
        old_to_consider = {
            x: y
            for x, y in old._column_provenance.items()
            if limit is None or limit.contains(x)
        }
        return [
            x
            for x, y in cols_to_consider.items()
            if x not in old_to_consider or y != old_to_consider[x]
        ]

    def copy(self):
        return TrackedColumns(
            # events=copy.copy(self._events),
            events=self._events,
            column_provenance=copy.copy(self._column_provenance),
            current_provenance=self._current_provenance,
            metadata=copy.copy(self.metadata),
            pipeline_data=copy.deepcopy(self.pipeline_data),
            backend=self.backend,
        )

    @staticmethod
    def fromEvents(events, metadata, backend, provenance: int):
        return TrackedColumns(
            events=events,
            column_provenance={
                x: provenance
                for x in getAllColumns(
                    events.layout, max_depth=events.layout.minmax_depth[1] - 2
                )
            },
            current_provenance=provenance,
            backend=backend,
            metadata=metadata,
        )

    def getKeyForColumns(self, columns):
        """
        Get an excecution key for the column.
        Returns a hash dependent on the provenance of all the columns contains in the input.
        """
        ret = []
        for column in columns:
            try:
                ret.append((column, self._column_provenance[column]))
            except KeyError as e:
                continue

        logger.debug(f"Relevant columns for {columns} are :\n {ret}")
        return hash((freeze(self.metadata), freeze(self.pipeline_data), freeze(ret)))

    def __setitem__(self, column, value):
        column = Column(column)
        if (
            self._allowed_outputs is not None
            and not TrackedColumns.INTERNAL_USE_COL.contains(column)
            and not self._allowed_outputs.contains(column)
        ):
            raise RuntimeError(
                f"Column {column} is not in the list of outputs {self._allowed_outputs}"
            )
        self._events = setColumn(self._events, column, value)
        self._column_provenance[column] = self._current_provenance
        if hasattr(value, "layout"):
            all_columns = getAllColumns(value.layout, column)
        else:
            all_columns = {column}
        for c in all_columns:
            self._column_provenance[c] = self._current_provenance
            # logger.debug(
            #     f"Adding column {c} to events with provenance {self._current_provenance}"
            # )
        for part in column.fields:
            c = Column(part)
            self._column_provenance[c] = self._current_provenance
            logger.debug(
                f"Updating parent column {c} to events with provenance {self._current_provenance}"
            )

        # for c in self._column_provenance:
        #     if c.contains(column):
        #         self._column_provenance[c] = self._current_provenance

    def __getitem__(self, column):
        column = Column(column)
        if self._allowed_inputs is not None and not self._allowed_inputs.contains(
            column
        ):
            raise RuntimeError(
                f"Column {column} is not in the list of inputs {self._allowed_inputs}"
            )
        return column.extract(self._events)

    def addColumnsFrom(self, other, columns):
        for column in columns:
            with self.useKey(other._column_provenance[column]):
                self[column] = other[column]
            # self._setIndividualColumnWithProvenance(
            #     column, other[column], other._column_provenance[column]
            # )

    def _addColumnsInternal(self, other, columns):
        for column in columns:
            self._events = setColumn(self._events, column, other[column])
            # self._setIndividualColumnWithProvenance(
            #     column, other[column], other._column_provenance[column]
            # )

    def filter(self, mask):
        if not self._allow_filter:
            raise RuntimeError()
        self._events = self._events[mask]
        for c in self._column_provenance:
            self._column_provenance[c] = self._current_provenance
        return self

    @contextlib.contextmanager
    def useKey(self, provenance):
        old_provenance = self._current_provenance
        self._current_provenance = provenance
        yield
        self._current_provenance = old_provenance

    @contextlib.contextmanager
    def allowedInputs(self, columns: list[Column] | ColumnCollection):
        if not isinstance(columns, ColumnCollection):
            columns = ColumnCollection(columns)
        old_inputs = self._allowed_inputs
        self._allowed_inputs = columns
        yield
        self._allowed_inputs = old_inputs

    @contextlib.contextmanager
    def allowedOutputs(self, columns: list[Column] | ColumnCollection):
        if not isinstance(columns, ColumnCollection):
            columns = ColumnCollection(columns)
        old_outputs = self._allowed_outputs
        self._allowed_outputs = columns
        yield
        self._allowed_outputs = old_outputs

    # @contextlib.contextmanager
    # def allowFilter(self, allow: bool):
    #     old_allow = self._allowed_filter
    #     self._allow_filter = allow
    #     yield
    #     self._allow_filter = old_allow


def mergeColumns(column_views):
    ret = copy.copy(column_views[0])
    for other in column_views[1:]:
        ret = ret.addColumnsFrom(other)
    return ret


def addSelection(columns, name, data):
    column = Column(("Selection", name))
    columns[column] = data
    if "Selections" not in columns.pipeline_data:
        columns.pipeline_data["Selections"] = {}
    columns.pipeline_data["Selections"][name] = False
