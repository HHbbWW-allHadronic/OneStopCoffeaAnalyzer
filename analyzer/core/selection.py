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

logger = logging.getLogger("analyzer.core")

class Cutflow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cutflow: list[tuple[str, Scalar]]
    one_cut: list[tuple[str, Scalar]]
    n_minus_one: list[tuple[str, Scalar]]

    def __add__(self, other):
        def add_tuples(a, b):
            return [(x, y) for x, y in accumulate([dict(a), dict(b)]).items()]

        return Cutflow(
            cutflow=add_tuples(self.cutflow, other.cutflow),
            one_cut=add_tuples(self.one_cut, other.one_cut),
            n_minus_one=add_tuples(self.n_minus_one, other.n_minus_one),
        )

    def concat(self, other):
        return Cutflow(
            cutflow=self.cutflow + other.cutflow,
            one_cut=self.one_cut + other.one_cut,
            n_minus_one=self.n_minus_one + other.n_minus_one,
        )

    @property
    def selection_efficiency(self):
        return self.cutflow[-1][1] / self.cutflow[0][1]


@dataclass
class SelectionSet:
    """
    Selection for a single sample.
    Stores the preselection and selection masks.
    The selection mask is relative to the "or" of all the preselection cuts.
    """

    selection: PackedSelection = field(default_factory=PackedSelection)

    parent_names: Optional[list[str]] = None
    parent: Optional["SelectionSet"] = None

    def __eq__(self, other):
        return (
            self.parent == other.parent
            and self.selection.names == other.selection.names
            and self.parent_names == other.parent_names
        )

    def allNames(self):
        ret = self.selection.names
        if self.parent is not None:
            ret += self.parent.names
        return ret
    
    def addMask(self, name, mask):
        if not name in self.allNames():
            self.selection.add(name, mask)

    def inclusiveMask(self):
        names = self.selection.names
        if not names:
            return None
        return self.selection.any(*names)

    def getMask(self, names):
        logger.info(f"Getting selection for names {names}")
        return self.selection.all(*names)

    def getCutflow(self, names):
        nmo = sel.nminusone(*names).result()
        cutflow = sel.cutflow(*names).result()
        onecut = list(map(tuple, zip(cutflow.labels, cutflow.nevonecut)))
        cumcuts = list(map(tuple, zip(cutflow.labels, cutflow.nevcutflow)))
        nmocuts = list(map(tuple, zip(nmo.labels, nmo.nev)))
        ret = Cutflow(cutflow=cumcuts, one_cut=onecut, n_minus_one=nmocuts)
        if parent is not None:
            parent_cutflow = parent.getCutflow(self.parent_names)
            ret = parent_cutflow + ret

        return ret


@dataclass
class Selection:
    select_from: SelectionSet
    names: tuple[str] = field(default_factory=tuple)

    def __add__(self, name):
        self.names = self.names + (name,)
        return self


    def getMask(self):
        return self.select_from.getMask(self.names)

    def __eq__(self, other):
        return self.select_from == other.select_from and self.names == other.names
