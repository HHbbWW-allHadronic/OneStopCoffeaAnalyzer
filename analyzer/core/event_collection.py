from __future__ import annotations

from rich import print
import random
import copy
from analyzer.core.columns import Column, TrackedColumns
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
from analyzer.core.caching import cache
import logging


logger = logging.getLogger("analyzer")


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
    logger.info(f'Getting replicas for dataset "{dataset}"')
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


@cache.memoize(tag="das-datasets")
def getDasFileSet(dataset):
    from coffea.dataset_tools import rucio_utils

    client = rucio_utils.get_rucio_client()
    return getReplicas(dataset, client)


def decideFile(possible_files, location_priorities=None):
    import re

    if location_priorities:

        def rank(val):
            x = list(
                (i for i, x in enumerate(location_priorities) if re.match(x, val)),
            )
            return next(iter(x), len(location_priorities))

        sites_ranked = [(rank(site), path) for path, site in possible_files.items()]
        max_rank = min(x[0] for x in sites_ranked)
        ret = random.choice([x[1] for x in sites_ranked if x[0] == max_rank])
        return ret
    else:
        return random.choice(list(possible_files.keys()))


@define
class FileInfo:
    file_path: str
    nevents: int | None = None
    tree_name: str | None = None
    schema_name: str | None = None
    chunks: set[tuple[int, int]] | None = None
    target_chunk_size: int | None = None

    def checkCompatible(self, other: FileInfo):
        if (
            self.target_chunk_size != other.target_chunk_size
            or self.nevents != other.nevents
            or self.schema_name != other.schema_name
            or self.tree_name != other.tree_name
        ):
            raise RuntimeError()

    def __iadd__(self, other: FileInfo):
        self.checkCompatible(other)
        oc = other.chunks or set()
        sc = self.chunks or set()
        self.chunks = oc | sc
        return self

    def __isub__(self, other: FileInfo):
        self.checkCompatible(other)
        oc = other.chunks or set()
        sc = self.chunks or set()
        self.chunks = sc - oc
        return self

    def __add__(self, other: FileInfo):
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def __sub__(self, other: FileInfo):
        ret = copy.deepcopy(self)
        ret -= other
        return ret

    def iChunk(self, chunk_size):
        if self.nevents is None or self.chunks is not None:
            raise RuntimeError()
        self.target_chunk_size = self.target_chunk_size or chunk_size
        self.chunks = chunkN(self.nevents, self.target_chunk_size)
        return self

    def chunked(self, chunk_size):
        ret = copy.deepcopy(self)
        ret.iChunk(chunk_size)
        return ret

    def intersects(self, other):
        self.checkCompatible(other)
        return bool(self.chunks.intersection(other.chunks))

    def intersection(self, other):
        self.checkCompatible(other)
        ret = copy.deepcopy(self)
        ret.chunks = self.chunks.intersection(other.chunks)
        return ret

    @property
    def chunked_events(self):
        if not self.chunks:
            return 0
        else:
            return sum(y - x for x, y in self.chunks)

    @property
    def empty(self):
        return not isinstance(self.chunks, None) and not self.chunks

    @property
    def is_chunked(self):
        return self.chunks is not None

    def toFileChunks(self):
        return [
            FileChunk(
                self.file_path,
                *c,
                self.tree_name,
                self.target_chunk_size,
                self.schema_name,
                self.nevents,
            )
            for c in self.chunks
        ]


def buildMissingFileset(
    source_description: SourceDescription, file_set: FileSet
) -> FileSet:
    """
    Two cases:
      1. The file never got preprocessed or none of the chunks ran. In this case, we add the empty file to the return set.
      2. The file was preprocesses but not all the chunks successfully ran. In this case, we compute the chunking based on the target chunk size and number of events, then take the difference.
    """
    missing_files = source_description.getMissing(file_set)
    fs = file_set.asMaximal()
    missing_chunks = fs - file_set
    return missing_files + missing_chunks


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
            files={
                x: FileInfo(x, tree_name=self.tree_name, schema_name=self.schema_name)
                for x in files
            },
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
            files={
                x: FileInfo(x, tree_name=self.tree_name, schema_name=self.schema_name)
                for x in files
            },
        )


@define
class FileListCollection(SourceDescription):
    files: list[str]
    tree_name: str = "Events"
    schema_name: str | None = None

    def getFileSet(self, **kwargs) -> FileSet:
        return FileSet(
            files={
                x: FileInfo(x, tree_name=self.tree_name, schema_name=self.schema_name)
                for x in self.files
            },
        )

    def getMissing(self, file_set: FileSet) -> FileSet:
        files = [x for x in self.files if x not in file_set.files]
        return FileSet(
            files={
                x: FileInfo(x, tree_name=self.tree_name, schema_name=self.schema_name)
                for x in files
            },
        )


def chunkN(nevents, chunk_size):
    nchunks = max(round(nevents / chunk_size), 1)
    chunk_size = math.ceil(nevents / nchunks)
    chunks = set(
        (
            i * chunk_size,
            min((i + 1) * chunk_size, nevents),
        )
        for i in range(nchunks)
    )
    return chunks


@define
class FileSet:
    files: dict[str, FileInfo]
    # chunk_size: int | None = None
    # tree_name: str = "Events"
    # schema_name: str | None = None

    # def checkCompatible(self, other: FileSet):
    #     if (
    #         self.chunk_size != other.chunk_size
    #         or self.tree_name != other.tree_name
    #         or self.schema_name != other.schema_name
    #     ):
    #         raise RuntimeError()

    @property
    def chunked_events(self):
        return sum(x.chunked_events for x in self.files.values())

    @property
    def total_file_events(self):
        return sum(x.nevents if x.nevents else 0 for x in self.files.values())

    @staticmethod
    def fromChunk(chunk):
        return FileSet(
            files={
                chunk.file_path: FileInfo(
                    file_path=chunk.file_path,
                    nevents=chunk.file_nevents,
                    chunks={(chunk.event_start, chunk.event_stop)},
                    target_chunk_size=chunk.target_chunk_size,
                    tree_name=chunk.tree_name,
                    schema_name=chunk.schema_name,
                )
            },
            # chunk_size=chunk.target_chunk_size,
        )

    def asMaximal(self):
        ret = copy.deepcopy(self)
        for file_info in ret.files.values():
            file_info.chunks = None
            file_info.iChunk(file_info.target_chunk_size)
        return ret

    def updateInfoFromOther(self, other: FileSet):
        if set(self.files) != set(other.files):
            raise RuntimeError
        for f in other:
            self[f].nevents = other[f].nevents
        return self

    def updateFileInfo(self, file_info):
        self.files[file_info.file_path] = file_info
        k = (file_info.file_path, file_info.tree_name)
        cache.set(k, file_info.nevents, tag="file-events")

    def getNeededUpdatesFuncs(self):
        return [
            ft.partial(getFileInfo, f.file_path, f.tree_name)
            for f in self.files.values()
            if f.nevents is None
        ]

    def updateFromCache(self):
        for finfo in self.files.values():
            k = (finfo.file_path, finfo.tree_name)
            if k in cache:
                nevents = cache[k]
                self.files[finfo.file_path].nevents = nevents

    def intersection(self, other):
        common_files = set(self.files).intersection(set(other.files))
        ret = {}
        for fname in common_files:
            ret[fname] = self.files[fname].intersection(other.files[fname])
        return FileSet(files=ret)

    def __iadd__(self, other):
        for fname, finfo_other in other.files.items():
            if fname not in self.files:
                self.files[fname] = copy.deepcopy(finfo_other)
            else:
                self.files[fname] += finfo_other
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
            steps_new = steps_self - steps_other
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
            x.chunks is not None and not x.chunks for x in self.files.values()
        )
        return ret

    def justChunked(self):
        ret = {}
        for k, v in self.files.items():
            if v.chunks is not None:
                ret[k] = v
        return FileSet(files=ret)

    def justUnchunked(self):
        ret = {}
        for k, v in self.files.items():
            if v.chunks is None:
                ret[k] = v
        return FileSet(files=ret)

    def applySliceChunks(self):
        ret = {}
        for k, v in self.files.items():
            if v is None:
                ret[k] = v
        return FileSet(files=ret)

    def splitFiles(self, files_per_set):
        lst = list(self.files.items())
        files_split = {(i, i + n): dict(lst[i : i + n]) for i in range(0, len(lst), n)}
        return {k: FileSet(files=v) for k, v in files_split.items()}

    def iterChunks(self):
        for f, v in self.files.items():
            if v.chunks is None:
                continue
            yield from iter(v.toFileChunks())

    def toChunked(self, chunk_size):
        files = {
            x: y.chunked(chunk_size) if y.chunks is None else y
            for x, y in self.files.items()
            if y.nevents is not None
        }
        return FileSet(files)


# @cache.memoize(tag="file-info")
def getFileInfo(file_path, tree_name, **kwargs):
    tree = uproot.open({file_path: None}, **kwargs)[tree_name]
    nevents = tree.num_entries
    return FileInfo(file_path, nevents, tree_name)


@define(frozen=True)
class FileChunk:
    file_path: str
    event_start: int | None = None
    event_stop: int | None = None
    tree_name: str = "Events"
    target_chunk_size: int | None = None
    schema_name: str | None = None
    file_nevents: int | None = None

    @property
    def nevents(self):
        if self.event_start is None or self.event_stop is None:
            return None
        return self.event_stop - self.event_start

    def loadEvents(self, backend="coffea-virtual", view_kwargs=None):
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

        return TrackedColumns.fromEvents(events, **view_kwargs)

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
