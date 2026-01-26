from analyzer.core.analysis_modules import AnalyzerModule
import uuid
import hashlib
from pathlib import Path
import uproot

from analyzer.utils.file_tools import copyFile
from analyzer.utils.structure_tools import dictToDot, dotFormat
import awkward as ak
from attrs import define, field


import logging


logger = logging.getLogger("analyzer.modules")


def isRootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        if isinstance(t.content, ak.types.NumpyType):
            return True
        if isinstance(t.content, ak.types.ListType) and isinstance(
            t.content.content, ak.types.NumpyType
        ):
            return True
    return False


def uprootWriteable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in events.fields:
        if events[bname].fields:
            out[bname] = ak.zip(
                {
                    n: ak.without_parameters(events[bname][n])
                    for n in events[bname].fields
                    if isRootcompat(events[bname][n])
                }
            )
        else:
            out[bname] = ak.to_packed(ak.without_parameters(events[bname]))
    return out


@define
class SaveEvents(AnalyzerModule):
    """
    Analyzer module that serializes and persists event-level data to ROOT files.

    `SaveEvents` writes the full event record from the current analysis columns
    to a ROOT file using `uproot`. Files are written locally first and then copied
    to a target destination defined by a configurable output path template.

    Each unique input (as defined by `getKeyNoParams`) is written at most once per
    process execution.

    Parameters
    ----------
    prefix : str
        Destination directory prefix where the output ROOT files will be copied.
        This may be a local or remote path, depending on the configured copy backend.

    output_format : str, optional
        Filename template used to construct the final output path. The template
        is expanded using metadata fields from `columns.metadata`, plus the
        following automatically provided fields:

        - ``file_id`` : MD5 hash of the input file path
        - ``uuid`` : Random UUID to guarantee local filename uniqueness

        Default:
        ``"{dataset_name}__{sample_name}__{file_id}__{chunk.event_start}_{chunk.event_stop}.root"``
    """

    prefix: str
    output_format: str = "{dataset_name}__{sample_name}__{file_id}__{chunk.event_start}_{chunk.event_stop}.root"
    __has_run: set = field(factory=set)

    def run(self, columns, params):
        k = self.getKeyNoParams(columns)
        if k in self.__has_run:
            return columns, []
        events = columns._events
        file_id = (
            hashlib.md5((columns.metadata["chunk"]["file_path"]).encode())
            .hexdigest()
            .upper()
        )
        uid = str(uuid.uuid4())

        target = dotFormat(
            self.output_format,
            **dict(dictToDot(columns.metadata)),
            file_id=file_id,
            uuid=uid,
        )

        target = self.prefix + "/" + target
        base = Path("localsaved")
        base.mkdir(exist_ok=True, parents=True)

        local_filename = base / f"{uid}.root"
        try:
            with uproot.recreate(local_filename, compression=uproot.ZSTD(5)) as f:
                f.mktree("Events", uprootWriteable(events))
            copyFile(local_filename, target)
        finally:
            local_filename.unlink(missing_ok=True)
        self.__has_run.add(k)
        return columns, []

    def inputs(self, metadata):
        return "EVENTS"

    def outputs(self, metadata):
        return []
