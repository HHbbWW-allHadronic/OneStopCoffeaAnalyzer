from __future__ import annotations

import uproot
import math

from attrs import define, field
from coffea.nanoevents import NanoAODSchema
from attrs import define, field, frozen

from collections.abc import Iterable


import abc
from typing import Any
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


@define
class SourceCollection(abc.ABC):
    @abc.abstractmethod
    def getSources(self) -> Iterable[EventSourceDescription]: ...

    @abc.abstractmethod
    def sub(self, sources: Iterable[EventSource]) -> EventCollection: ...

class DasCollection(SourceCollection):
    das_path: str
    tree_name: str = "Events"
    schema_name: str | None = None

    metadata: str | dict[str, Any] | None = None

    _files: list[set[str]] | None = None


@frozen
class EventSourceDescription(abc.ABC):
    @abc.abstractmethod
    def materialize(self): ...

@frozen
class EventSource(abc.ABC):
    @abc.abstractmethod
    def loadEvents(
        self, backend, metadata, event_start=None, event_stop=None, view_kwargs=None
    ) -> ColumnView: ...

    @abc.abstractmethod
    def chunks(self, chunk_size, max_chunks=None) -> Iterable[SourceChunk]: ...


def getFileEvents(file_path, tree_name):
    tree = uproot.open({file_path: None}, **kwargs)[tree_name]
    nevents = tree.num_entries
    return nevents


@define(frozen=True)
class FileSource(EventSource):
    file_path: str
    tree_name: str = "Events"
    schema_name: str | None = None

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

    def loadEvents(self, backend, metadata, start=None, stop=None, view_kwargs=None):
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



def getFilesDas(das_path):
    pass




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
            yield SourceChunk(
                source_file=self.source, event_start=start, event_stop=stop
            )

    def loadEvents(self, backend, metadata, view_kwargs=None):
        return self.source_file.loadEvents(
            backend,
            metadata,
            start=self.event_start,
            stop=self.event_stop,
            view_kwargs=view_kwargs,
        )

    def overlaps(self, other):
        same_source = self.source == other.source
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
