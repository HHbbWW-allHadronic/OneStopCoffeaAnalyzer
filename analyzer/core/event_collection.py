from __future__ import annotations

import random
from analyzer.core.columns import Column, ColumnView
from cattrs.strategies import include_subclasses, configure_tagged_union
import functools as ft

import uproot
import math

from attrs import define, field
from coffea.nanoevents import NanoAODSchema
from attrs import define, field, frozen

from collections.abc import Iterable


import abc
from typing import Any
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from analyzer.core.caching import makeCached



@define
class SourceDescription(abc.ABC):
    @abc.abstractmethod
    def getFileSet(self) -> FileSet: ...

    @abc.abstractmethod
    def getMissing(self, fs: FileSet, **kwargs) -> FileSet: ...


def getDatasets(query, client):
    from coffea.dataset_tools import rucio_utils

    outlist, outtree = rucio_utils.query_dataset(
        query,
        client=client,
        tree=True,
        scope="cms",
    )
    return outlist


def getReplicas(dataset, client):
    from analyzer.utils.file_tools import extractCmsLocation
    from coffea.dataset_tools import rucio_utils

    (
        outfiles,
        outsites,
        sites_counts,
    ) = rucio_utils.get_dataset_files_replicas(
        dataset,
        allowlist_sites=[],
        blocklist_sites=["T3_CH_CERN_OpenData"],
        regex_sites=[],
        mode="full",
        client=client,
    )
    ret = [dict(zip(s, f)) for s, f in zip(outfiles, outsites)]
    return ret


@makeCached()
def getDasFileSet(dataset):
    from coffea.dataset_tools import rucio_utils

    client = rucio_utils.get_rucio_client()
    return getReplicas(dataset, client)


def decideFile(possible_files, location_priorities=None):
    if location_priorities:

        def rank(val):
            x = list(
                (i for i, x in enumerate(location_priority_regex) if re.match(x, val)),
            )
            return next(iter(x), len(location_priority_regex))

        sites_ranked = [(rank(site), path) for path, site in possible_files.items()]
        max_rank = max(x[0] for x in sites_ranked)
        return random.choice([x[1] for x in sites_ranked if x[0] == max_rank])
    else:
        return random.choice(list(possible_files.keys()))


@define
class FileInfo:
    nevents: int | None = None
    chunks: list[tuple[int, int]] | None = None


@define
class DasCollection(SourceDescription):
    das_path: str
    tree_name: str = "Events"
    schema_name: str | None = None

    def getFileSet(self, location_priorities=None) -> FileSet:
        all_files = getDasFileSet(self.das_path)
        files = [
            decideFile(x, location_priorities=location_priorities) for x in all_files
        ]
        return FileSet(
            files={x: FileInfo() for x in files},
            chunk_size=None,
            tree_name=self.tree_name,
            schema_name=self.schema_name,
        )

    def getMissing(self, file_set: FileSet, location_priorities=None):
        all_files = getDasFileSet(self.das_path)
        fs_files = set(file_set.files)
        files = [
            decideFile(x, location_priorities)
            for x in all_files
            if not any(y in fs_files for y in x.keys())
        ]
        return FileSet(
            files={x: FileInfo() for x in files},
            chunk_size=file_set.chunk_size,
            tree_name=file_set.tree_name,
            schema_name=file_set.schema_name,
        )


@define
class FileListCollection(SourceDescription):
    files: list[str]
    tree_name: str = "Events"
    schema_name: str | None = None

    def getFileSet(self, **kwargs) -> FileSet:
        return FileSet(
            files={x: FileInfo() for x in self.files},
            chunk_size=None,
            tree_name=self.tree_name,
            schema_name=self.schema_name,
        )

    def getMissing(self, file_set: FileSet) -> FileSet:
        files = [x for x in self.files if x not in self.file_set.files]
        return FileSet(
            files={x: FileInfo() for x in files},
            chunk_size=file_set.chunk_size,
            tree_name=file_set.tree_name,
            schema_name=file_set.schema_name,
        )


def chunkN(nevents, chunk_size):
    nchunks = max(round(nevents / chunk_size), 1)
    chunk_size = math.ceil(nevents / nchunks)
    chunks = [
        [
            i * chunk_size,
            min((i + 1) * chunk_size, nevents),
        ]
        for i in range(nchunks)
    ]
    return chunks


@define
class FileSet:
    files: dict[str, FileInfo]
    chunk_size: int | None = None
    tree_name: str = "Events"
    schema_name: str | None = None

    @staticmethod
    def fromChunk(chunk):
        return FileSet(
            files={chunk.file_path: [(chunk.event_start, chunk.event_stop)]},
            tree_name=chunk.tree_name,
            schema_name=chunk.schema_name,
        )

    def updateEvents(self, fname, events):
        self.files[fname].nevents = events

    def getNeededUpdatesFuncs(self):
        return [
            ft.partial(getFileEvents, f, self.tree_name)
            for f, v in self.files.items()
            if v.nevents is None
        ]

    def intersect(self, other):
        common_files = set(self.files).intersection(set(other.files))
        ret = {}
        for fname in common_files:
            steps_self, steps_other = (
                self.files[fname].chunks,
                other.files[fname].chunks,
            )
            if not (data_this["steps"] is None and data_other["steps"] is None):
                data["steps"] = [
                    s for s in (steps_self or []) if s in (steps_other or [])
                ]
            ret[fname].nevents = self.files[fname].nevents
            ret[fname].chunks = data

        return FileSet(files=ret, chunk_size=self.chunk_size)

    def __iadd__(self, other):
        for fname, steps_other in other.files.items():
            if fname not in self.files:
                self.files[fname] = steps_other
            else:
                if steps_self is None and steps_other is None:
                    steps_new = None
                else:
                    steps_self = set(self.files[fname].chunks or [])
                    steps_other = set(steps_other.chunks or [])
                    steps_new = list(sorted(steps_self | steps_other))
                self.files[fname] = steps_new
        return self

    def __isub__(self, other):
        common_files = set(self.files).intersection(other.files)

        for fname in common_files:
            if self.files[fname] is None != other.files[fname] is None:
                raise RuntimeError()
            if self.files[fname] is None and other.files[fname] is None:
                del self.files[fname]

            steps_self = set(self.files[fname].chunks)
            steps_other = set(other.files[fname].chunks)
            steps_new = list(sorted(steps_self - steps_other))
            self.files[fname].chunks = steps_new

        return self

    def __add__(self, other):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def __sub__(self, other):
        ret = copy.deepcopy(self)
        ret -= other
        return ret

    @property
    def materialized(self):
        return not any(x is None for x in self.files.values())

    @property
    def empty(self):
        ret = not self.files or all(
            x is not None and not x for x in self.files.values()
        )
        return ret

    def justChunked(self):
        ret = {}
        for k, v in self.files.items():
            if v is not None:
                ret[k] = v
        return FileSet(files=ret, chunk_size=self.chunk_size)

    def justUnchunked(self):
        ret = {}
        for k, v in self.files.items():
            if v is None:
                ret[k] = v
        return FileSet(files=ret, chunk_size=self.chunk_size)

    def applySliceChunks(self):
        ret = {}
        for k, v in self.files.items():
            if v is None:
                ret[k] = v
        return FileSet(files=ret, chunk_size=self.chunk_size)

    def splitFiles(self, files_per_set):
        lst = list(self.files.items())
        files_split = {(i, i + n): dict(lst[i : i + n]) for i in range(0, len(lst), n)}
        return {
            k: FileSet(
                files=v,
                chunk_size=self.chunk_size,
            )
            for k, v in files_split.items()
        }

    def iterChunks(self):
        for f, v in self.files.items():
            if v.chunks is None:
                continue
            for chunk in v.chunks:
                yield FileChunk(f, *chunk, self.tree_name, self.schema_name)

    def toChunked(self, chunk_size):
        files = {
            x: FileInfo(y.nevents, chunkN(y.nevents, chunk_size))
            for x, y in self.files.items()
            if y.nevents is not None
        }
        return FileSet(files, chunk_size, self.tree_name, self.schema_name)


@makeCached()
def getFileEvents(file_path, tree_name, **kwargs):
    tree = uproot.open({file_path: None}, **kwargs)[tree_name]
    nevents = tree.num_entries
    return file_path, nevents


@define(frozen=True)
class FileChunk:
    file_path: str
    event_start: int | None = None
    event_stop: int | None = None
    tree_name: str = "Events"
    schema_name: str | None = None

    @property
    def metadata(self):
        return self.source.metadata

    def loadEvents(self, backend, view_kwargs=None):
        view_kwargs = view_kwargs or {}
        view_kwargs["backend"] = backend
        start = self.event_start
        stop = self.event_stop
        if backend == "coffea-virtual":
            events = NanoEventsFactory.from_root(
                {self.file_path: self.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="virtual",
            ).events()
        elif backend == "coffea-dask":
            events = NanoEventsFactory.from_root(
                {self.file_path: self.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="dask",
            ).events()
        elif backend == "coffea-eager":
            events = NanoEventsFactory.from_root(
                {self.file_path: self.tree_name},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="eager",
            ).events()
        elif backend == "RDF":
            pass

        return ColumnView.fromEvents(events, **view_kwargs)

    def overlaps(self, other: FileChunk):
        same_source = self.file_path == other.file_path
        if not same_file:
            return False
        return (
            self.event_start <= other.event_stop
            and other.event_start <= self.event_stop
        )

    def toFileSet(self):
        return FileSet.fromChunk(self)


def configureConverter(conv):
    # union_strategy = ft.partial(configure_tagged_union, tag_name="module_name")
    include_subclasses(SourceDescription, conv)  # , union_strategy=union_strategy)
