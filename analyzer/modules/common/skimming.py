from analyzer.core.analysis_modules import AnalyzerModule, register_module
import uuid
import hashlib
from pathlib import Path
import tempfile
import re
import uproot

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.core.results import SavedEventFile, SavedFiles
from analyzer.utils.file_tools import copyFile
from analyzer.utils.structure_tools import flatten, dictToDot, doFormatting
import uuid
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import itertools as it
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
import copy
from rich import print
import numpy as np


import awkward as ak
import correctionlib
import pydantic as pyd
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import correctionlib.schemav2 as cs
from functools import lru_cache
import logging


from analyzer.core.analysis_modules import (
    AnalyzerModule,
    register_module,
    MetadataExpr,
    MetadataAnd,
    IsRun,
    IsSampleType,
)

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
    prefix: str
    output_format: str = "{dataset_name}__{sample_name}__{file_id}__{chunk.event_start}_{chunk.event_stop}"

    def run(self, columns, params):
        events = columns._events
        file_id = (
            hashlib.md5((columns.metadata["chunk"]["file_path"]).encode())
            .hexdigest()
            .upper()
        )
        uid = str(uuid.uuid4())

        target = doFormatting(
            self.output_format,
            **dict(dictToDot(columns.metadata)),
            file_id=file_id,
            uuid=uid,
        )

        target = self.prefix + target
        base = Path("localsaved")
        base.mkdir(exist_ok=True, parents=True)

        local_filename = base / f"{uid}.root"
        try:
            with uproot.recreate(local_filename, compression=uproot.ZSTD(5)) as f:
                f["Events"] = uprootWriteable(events)
            copyFile(local_filename, target)
        finally:
            local_filename.unlink(missing_ok=True)
        return columns, []

    def inputs(self, metadata):
        return "EVENTS"

    def outputs(self, metadata):
        return []
