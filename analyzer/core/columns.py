import concurrent.futures
import copy
import enum
import inspect
import itertools as it
import logging
import pickle as pkl
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import yaml

import awkward as ak
import dask
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId, SampleType
from analyzer.utils.file_tools import extractCmsLocation
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import decompress_form
from pydantic import BaseModel, ConfigDict, Field

from .analysis_modules import MODULE_REPO, AnalyzerModule, ModuleType
from .common_types import Scalar
from .specifiers import SampleSpec, SubSectorId

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track

SHAPE_VAR_SEPARATOR="__"

logger=logging.getLogger(__name__)

@dataclass
class Column:
    name: str
    shape_variations: list[ str] = field(default_factory=list)


    def getColumnName(self, shape_var=None):

        if shape_var is None:
            return self.name
        return self.name + SHAPE_VAR_SEPARATOR + shape_var

@dataclass
class Columns:
    events: Any
    columns: dict[str, Column] = field(default_factory=dict)
    base: Optional["Columns"] = None
    sys: Optional[tuple[str,str]]= None


    def __getattr__(self, attr):
        return getattr(self.events, attr)

    def __iter__(self):
        return iter(self.events.fields)

    def colnames(self):
        return list(events.fields)

    def allShapes(self):
        return list(it.chain.from_iterable([(x.name, y) for y in x.shape_variations]  for x in self.columns.values()))

    def add(self, name, nominal_value, variations=None):
        

        logger.info(f'Adding columns {name} with variations: {list(variations)}')
        col = Column(
            name=name, shape_variations=list(variations)
        )
        self.columns[name] = col
        self.events[col.name] = nominal_value
        for svn, val in variations.items():
            self.events[col.getColumnName(svn)]  = val



    def get(self, name, variation=None):
        col = self.columns[name]
        n = col.getColumnName(variation)
        logger.info(f'Getting column "{name}" with variation "{variation}" = "{n}"')
        return self.events[n]



class ColumnShapeSyst(Columns):
    def __init__(self, base, syst=None):
        self.this = Columns(self.base.events)
        self.syst = syst



    def __iter__(self):
        return it.chain(iter(self.this), iter(self.base))
