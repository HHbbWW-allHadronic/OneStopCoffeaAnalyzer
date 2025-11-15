from __future__ import annotations

from attrs import define, field


import contextlib
import functools as ft
import copy
import awkward as ak
from typing import Any
import awkward as ak
import logging
from analyzer.utils.structure_tools import freeze

logger = logging.getLogger("analyzer.core")

def coerceFields(data):
    if isinstance(data, str):
        return tuple(data.split("."))
    elif isinstance(data, Column):
        return data.fields
    else:
        return data


@define(frozen=True)
class Column:
    fields: tuple[str, ...] = field(converter=coerceFields)

    def parts(self):
        return tuple(self.path)

    def contains(self, other):
        if len(self) > len(other):
            return False
        return other[: len(self)] == self

    def commonParent(self, other):
        l = []
        for x, y in zip(self.iterParts(), other.iterParts()):
            if x == y:
                l.append(x)
            else:
                break
        ret = Column(tuple(l))
        return ret

    def extract(self, events):
        return ft.reduce(lambda x, y: x[y], self.fields, events)

    def iterParts(self):
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

    def __iter__(self):
        return (Column(x) for x in self.fields)

    def __hash__(self):
        return hash(self.fields)

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
    columns: set[Column]

    def __iter__(self):
        return iter(columns)

    def contains(self, other: Column):
        return any(x.contains(other) for x in self.columns)

    def intersect(self, other: ColumnCollection):
        ret = {
            x
            for x in self.columns
            if any((x.contains(o) or o.contains(x)) for o in other)
        }


def getAllColumns(events, cur_col=None):
    if fields := getattr(events, "fields"):
        ret = set()
        for f in fields:
            if cur_col is not None:
                n = cur_col + f
            else:
                n = Column(f)
            ret |= getAllColumns(events[f], n)

        if cur_col is not None:
            ret.add(cur_col)
        return ret
    else:
        return {cur_col}


@define
class ColumnView:
    INTERNAL_USE_COL: ClassVar[Column] = Column("INTERNAL_USE")
    _events: Any
    _column_provenance: dict[Column, int]
    backend: EventBackend
    _current_provenance: int | None = None
    _allowed_inputs: ColumnCollection | None = None
    _allowed_outputs: ColumnCollection | None = None
    _allow_filter: bool = True
    metadata: Any | None = None
    pipeline_data: dict[str, Any] = field(factory=dict)

    @property
    def fields(self):
        return self._events.fields

    def updatedColumns(self, old, limit):
        cols_to_consider = {
            x: y for x, y in self._column_provenance.items() if limit.contains(x)
        }
        old_to_consider = {
            x: y for x, y in old._column_provenance.items() if limit.contains(x)
        }
        return [
            x
            for x, y in cols_to_consider.items()
            if x not in old_to_consider or y != old_to_consider[x]
        ]

    def copy(self):
        return ColumnView(
            events=copy.copy(self._events),
            column_provenance=copy.copy(self._column_provenance),
            metadata=copy.copy(self.metadata),
            pipeline_data=copy.deepcopy(self.pipeline_data),
            backend=self.backend,
        )

    @staticmethod
    def fromEvents(events, metadata, backend, provenance):
        return ColumnView(
            events=events,
            column_provenance={x: provenance for x in getAllColumns(events)},
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
            for c, v in self._column_provenance.items():
                if column.contains(c):
                    ret.append((c, v))

        logger.info(f"Relevant columns for {columns} are :\n {ret}")
        return hash((freeze(self.metadata), freeze(self.pipeline_data), freeze(ret)))

    def __setitem__(self, column, value):
        column = Column(column)
        if (
            self._allowed_outputs is not None
            and not ColumnView.INTERNAL_USE_COL.contains(column)
            and not self._allowed_outputs.contains(column)
        ):
            raise RuntimeError(
                f"Column {column} is not in the list of outputs {self._allowed_outputs}"
            )
        self._events = setColumn(self._events, column, value)
        self._column_provenance[column] = self._current_provenance
        all_columns = getAllColumns(value, column)
        for c in all_columns:
            self._column_provenance[c] = self._current_provenance
            logger.info(
                f"Adding column {c} to events with provenance {self._current_provenance}"
            )
        for c in self._column_provenance:
            if c.contains(column):
                self._column_provenance[c] = self._current_provenance
                logger.info(
                    f"Updating parent column {c} to events with provenance {self._current_provenance}"
                )

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

    def filter(self, mask):
        if not self._allow_filter:
            raise RuntimeError()
        new_cols = copy.copy(self)
        new_cols._events = new_cols._events[mask]
        for c in self._column_provenance:
            self._column_provenance[c] = self._current_provenance
        return new_cols

    @contextlib.contextmanager
    def useKey(self, provenance: ModuleProvenance):
        old_provenance = self._current_provenance
        self._current_provenance = provenance
        yield
        self._current_provenance = old_provenance

    @contextlib.contextmanager
    def allowedInputs(self, columns):
        columns = ColumnCollection(columns)
        old_inputs = self._allowed_inputs
        self._allowed_inputs = columns
        yield
        self._allowed_inputs = old_inputs

    @contextlib.contextmanager
    def allowedOutputs(self, columns):
        columns = ColumnCollection(columns)
        old_outputs = self._allowed_outputs
        self._allowed_outputs = columns
        yield
        self._allowed_outputs = old_outputs

    @contextlib.contextmanager
    def allowFilter(self, allow):
        old_allow = self._allowed_filter
        self._allow_filter = allow
        yield
        self._allow_filter = old_allow


def mergeColumns(column_views):
    ret = copy.copy(column_views[0])
    for other in column_views[1:]:
        ret = ret.addColumnsFrom(other)
    return ret
