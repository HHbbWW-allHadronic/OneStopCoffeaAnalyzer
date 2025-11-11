from __future__ import annotations
import numpy as np
import cProfile, pstats, io

import timeit
import uproot
from superintervals import IntervalMap
import enum
import math
import random

import numbers
import itertools as it
import dask_awkward as dak
import hist
from attrs import asdict, define, make_class, Factory, field
from cattrs import structure, unstructure, Converter
import hist
from coffea.nanoevents import NanoAODSchema
from attrs import asdict, define, make_class, Factory, field
import cattrs
from cattrs import structure, unstructure, Converter
from cattrs.strategies import include_subclasses, configure_tagged_union
import cattrs
from attrs import make_class

from collections.abc import Collection, Iterable
from collections import deque, defaultdict

import contextlib
import uuid
import functools as ft

from rich import print
import copy
import dask
import abc
import awkward as ak
from typing import Any, Literal
from functools import cached_property
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import logging
from rich.logging import RichHandler

@define
class SourceCollection(abc.ABC):
    nevents: int

    @abc.abstractmethod
    def getSources(self) -> Iterable[EventSource]: ...

    @abc.abstractmethod
    def sub(self, sources: Iterable[EventSource]) -> EventCollection: ...


@define
class EventSource(abc.ABC):
    @abc.abstractmethod
    def loadEvents(
        self, backend, metadata, event_start=None, event_stop=None, view_kwargs=None
    ) -> ColumnView: ...

    @abc.abstractmethod
    def chunks(self, chunk_size, max_chunks=None) -> Iterable[SourceChunk]: ...

    # @abc.abstractmethod
    # def intersect(self, other) -> EventSource: ...
    #
    # @abc.abstractmethod
    # def add(self, other) -> EventSource: ...
    #
    # @abc.abstractmethod
    # def sub(self, other) -> EventSource: ...


def getFileEvents(file_path, tree_name):
    tree = uproot.open({file_path: None}, **kwargs)[tree_name]
    nevents = tree.num_entries
    return nevents


@define(frozen=True)
class FileSource(EventSource):
    file_path: str
    tree_name: str = "Events"
    schema_name: str | None = None
    metadata: str | dict[str, Any] | None = None

    _nevents: int | None = field(default=None, eq=False)

    def getNumEvents(self, **kwargs):
        if self._nevents is not none:
            return self._nevents
        self._nevents = getFileEvents(self.file_path, self.tree_name)
        return nevents

    def chunks(self, chunk_size, max_chunks=None, exec_function=None, **kwargs):
        nevents = self.getNumEvents(**kwargs)
        nchunks = max(round(nevents / chunk_size), 1)
        chunk_size = math.ceil(nevents / nchunks)
        chunks = [
            [
                i * chunk_size,
                min((i + 1) * chunk_size, nevents),
            ]
            for i in range(nchunks)
        ]
        for start, stop in chunks:
            yield ChunkedRootFile(source_file=self, event_start=start, event_stop=stop)

    def loadEvents(self, backend, start=None, stop=None, view_kwargs=None):
        view_kwargs = view_kwargs or {}
        view_kwargs["backend"] = backend
        if backend == "coffea-virtual":
            events = NanoEventsFactory.from_root(
                {self.source_file.file_path: self.source_file.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="virtual",
            ).events()
        elif backend == "coffea-dask":
            events = NanoEventsFactory.from_root(
                {self.source_file.file_path: self.source_file.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="dask",
            ).events()
        elif backend == "coffea-eager":
            events = NanoEventsFactory.from_root(
                {self.source_file.file_path: self.source_file.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="eager",
            ).events()
        elif backend == "RDF":
            pass

        return ColumnView.fromEvents(events, **view_kwargs)


class Sample:
    sample_name: str
    x_sec: float | None = None
    event_source: EventCollection





class Dataset:
    dataset_name: str
    era: str
    data_type: DataType
    samples: list[Sample] = field(factory=list)


def getFilesDas(das_path):
    pass

class DasCollection(EventCollection):
    das_path: str
    tree_name: str = "Events"
    schema_name: str | None = None

    metadata: str | dict[str, Any] | None = None

    _files: list[RootFile] | None = None

    def preprocess(self):
        self._files = getFilesDas(self.das_path)


@define(frozen=True)
class SourceChunk(EventSource):
    source: EventSource
    event_start: int
    event_stop: int

    @property
    def metadata(self):
        return self.source.metadata

    def chunks(self, chunk_size, max_chunks=None, **kwargs):
        nevents = end - start
        nchunks = max(round(nevents / chunk_size), 1)
        chunk_size = math.ceil(nevents / nchunks)
        chunks = [
            [
                i * chunk_size,
                min((i + 1) * chunk_size, nevents),
            ]
            for i in range(nchunks)
        ]
        for start, stop in chunks:
            yield ChunkedRootFile(source_file=self, event_start=start, event_stop=stop)

    def loadEvents(self, backend, view_kwargs=None):
        return self.source_file.loadEvents(
            backend,
            start=self.event_start,
            stop=self.event_stop,
            view_kwargs=view_kwargs,
        )

    def overlaps(self, other):
        same_file = self.source_file.file_id == other.source_file.file_id
        if not same_file:
            return False
        return (
            self.event_start <= other.event_stop
            and other.event_start <= self.event_stop
        )


@define
class EventMultiSet(EventCollection):
    events_collections: list[EventCollection]

    def iterChunks(self, chunk_size):
        pass
