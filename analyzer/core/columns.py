from __future__ import annotations
import copy
import itertools as it
import logging
from dataclasses import dataclass, field
from analyzer.utils.debugging import jumpIn
from typing import Any
import dask_awkward as dak
from .selection import Selection, SelectionSet
from functools import lru_cache


logger = logging.getLogger(__name__)

SHAPE_VAR_SEPARATOR = "__"


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
    base: Columns | None = None
    syst: tuple[str, str] | None = None

    __column_cache: dict[tuple[str, tuple[str, str] | None], Any] = field(
        default_factory=dict
    )

    parent_columns: Columns | None = None
    parent_selection: Selection | None = None
    selection: SelectionSet | None = None

    # def __hash__(self):
    #     return hash((self.events.name, tuple(self.columns), self.syst))
    @lru_cache
    def filterBySelector(self, names):
        new_events = self.selection.all(*names)
        return Columns(
            new_events,
            parent_columns=self,
            columns=self.colummns,
            base=self.base,
            selection=SelectionSet(parent_selection=self.parent_selection),
            parent_selection=Selection(
                select_from=self.parent_selection, names=tuple(name)
            ),
        )

    def addSelectorMask(self, name, mask):
        self.selection.addMask(name, mask)

    @property
    def delayed(self):
        return isinstance(self.events, dak.Array)

    def __getattr__(self, attr):
        if attr in self.columns:
            return self.get(attr)
        return getattr(self.events, attr)

    def __getitem__(self, item):
        if item in self.columns:
            return self.get(item)
        return getattr(self.events, item)

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
            self.columns[name] = col
        col = self.columns[name]
        if syst is not None and syst not in col.shape_variations:
            col.shape_variations.append(syst)

        cname = col.getColumnName(syst)
        if cname not in self.events.fields:
            logger.debug(f"Adding column to events: {cname}")
            self.events[cname] = value
        # jumpIn(**locals())

    def add(self, name, nominal_value, variations=None, shape_dependent=False):
        if self.syst is not None and variations is not None:
            raise RuntimeError()

        variations = variations or {}

        if self.syst is not None:
            if shape_dependent:
                self.addVariation(
                    name, nominal_value, syst=self.syst[0] + "__" + self.syst[1]
                )
            else:
                self.addVariation(name, nominal_value)
        else:
            self.addVariation(name, nominal_value)
            for syst, val in variations.items():
                self.addVariation(name, val, syst=syst)

    def get(self, name):
        # if (name,self.syst) in self.__column_cache:
        #     return self.__column_cache[(name,self.syst)]
        if name in self.columns:
            col = self.columns[name]
            if self.syst is None:
                n = name
            elif self.syst and name == self.syst[0]:
                n = col.getColumnName(self.syst[1])
            else:
                real_syst = "__".join(self.syst)
                if real_syst in col.shape_variations:
                    n = col.getColumnName(real_syst)
                else:
                    n = col.getColumnName()
        else:
            n = name
        logger.debug(
            'Getting column "%s" with variation "%s" = "%s"', name, self.syst, n
        )
        # self.__column_cache[(name,self.syst)] = self.events[n]
        return self.events[n]

    def withSyst(self, syst):
        return Columns(self.events, copy.deepcopy(self.columns), self.base, syst)

    def withEvents(self, events):
        return Columns(events, copy.deepcopy(self.columns), self.base, self.syst)

    @property
    def fields(self):
        return self.events.fields
