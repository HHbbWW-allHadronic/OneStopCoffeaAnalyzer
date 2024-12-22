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

SHAPE_VAR_SEPARATOR = "__"

logger = logging.getLogger(__name__)


@dataclass
class Column:
    name: str
    shape_variations: list[str] = field(default_factory=list)

    def getColumnName(self, shape_var=None):
        if shape_var is None:
            return self.name
        return self.name + SHAPE_VAR_SEPARATOR + shape_var


@dataclass
class Columns:
    events: Any
    columns: dict[str, Column] = field(default_factory=dict)
    base: Optional["Columns"] = None
    syst: Optional[tuple[str, str]] = None

    def __getattr__(self, attr):
        if attr in self.columns:
            return self.get(attr)
        return getattr(self.events, attr)

    def __iter__(self):
        return iter(self.events.fields)

    def colnames(self):
        return list(events.fields)

    def allShapes(self):
        return list(
            it.chain.from_iterable(
                [(x.name, y) for y in x.shape_variations] for x in self.columns.values()
            )
        )


    def getSystName(self):
        if self.syst is not None:
            return self.syst[1]
        return None

    def getSystColumn(self):
        if self.syst is not None:
            return self.syst[0]
        return None


    def addVariation(self, name, value, syst=None):
        if name not in self.columns:
            col = Column(name=name)
            self.columns[name]=col
        col = self.columns[name]
        if syst is not None and syst not in col.shape_variations:
            col.shape_variations.append(syst)

        cname = col.getColumnName(syst)
        if cname not in self.events.fields:
            logger.info(f'Adding column to events: "{cname}"')
            self.events[cname] = value
            


    def add(self, name, nominal_value, variations=None, shape_dependent=False):
        if self.syst is not None and variations is not None:
            raise RuntimeError()

        variations = variations or {}
        logger.info(f"Adding columns {name} with variations: {list(variations)}")


        if self.syst is not None:
            if shape_dependent:
                self.addVariation(name, nominal_value, syst=self.syst[0] + "__" + self.syst[1])
            else:
                self.addVariation(name, nominal_value)
        else:
            self.addVariation(name, nominal_value)
            for syst,val in variations.items():
                self.addVariation(name, val, syst=syst)

    def get(self, name):
        if name in self.columns:
            col = self.columns[name]
            if self.syst is None:
                n = name
            elif self.syst and name == self.syst[0]:
                n = col.getColumnName(self.syst[1])
            else:
                real_syst =  "__".join(self.syst)
                if real_syst in col.shape_variations:
                    n = col.getColumnName(real_syst)
                else:
                    n = col.getColumnName()
        else:
            n = name
        logger.info(f'Getting column "{name}" with variation "{self.syst}" = "{n}"')
        return self.events[n]

    def withSyst(self, syst):
        return Columns(self.events, copy.deepcopy(self.columns), self.base, syst)

    def withEvents(self, events):
        return Columns(events, copy.deepcopy(self.columns), self.base, self.syst)
