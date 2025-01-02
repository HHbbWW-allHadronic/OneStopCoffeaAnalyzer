import copy
import logging
from dataclasses import dataclass, field


from analyzer.configuration import CONFIG
from coffea.analysis_tools import PackedSelection
from pydantic import BaseModel, ConfigDict

from .common_types import Scalar
from analyzer.utils.structure_tools import accumulate

if CONFIG.PRETTY_MODE:
    pass

logger = logging.getLogger("analyzer.core")

class SelectionFlow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cutflow: list[tuple[str, Scalar]]
    one_cut: list[tuple[str, Scalar]]
    n_minus_one: list[tuple[str, Scalar]]


    @property
    def total_events(self):
        return cutflow[0][1]

    @property
    def selection_efficiency(self):
        return self.cutflow[-1][1] / self.cutflow[0][1]

    def total_events(self):
        return cutflow[0][1]

    def __add__(self, other):
        def add_tuples(a, b):
            return [(x, y) for x, y in accumulate([dict(a), dict(b)]).items()]

        return SelectionFlow(
            cutflow=add_tuples(self.cutflow, other.cutflow),
            one_cut=add_tuples(self.one_cut, other.one_cut),
            n_minus_one=add_tuples(self.n_minus_one, other.n_minus_one),
        )

    def concatChild(self, child):
        def dropDup(flow):
            seen = set()
            return [x for x in flow if not (x[0] in seen or seen.add(x[0]))]
        return SelectionFlow(
            cutflow=dropDup(self.cutflow + child.cutflow),
            one_cut=self.one_cut + child.one_cut,
            n_minus_one=self.n_minus_one + child.n_minus_one,
        )



@dataclass
class SelectionSet:
    """
    Selection for a single sample.
    Stores the preselection and selection masks.
    The selection mask is relative to the "or" of all the preselection cuts.
    """

    selection: PackedSelection = field(default_factory=lambda: PackedSelection())
    parent_selection: "Selection" = None

    def __eq__(self, other):
        return (
            self.parent == other.parent
            and self.selection.names == other.selection.names
            and self.parent_names == other.parent_names
        )

    def allNames(self):
        ret = copy.copy(self.selection.names)
        if self.parent_selection is not None:
            ret += self.parent_selection.select_from.allNames()
        return ret
    
    def addMask(self, name, mask):
        if name not in self.allNames():
            self.selection.add(name, mask)


    def inclusiveMask(self):
        names = self.selection.names
        if not names:
            return None
        return self.selection.any(*names)

    def getMask(self, names):
        logger.info(f"Getting selection for names {names}")
        if names:
            return self.selection.all(*names)
        else:
            return None

    def getSelectionFlow(self, names):
        nmo = self.selection.nminusone(*names).result()
        cutflow = self.selection.cutflow(*names).result()
        onecut = list(map(tuple, zip(cutflow.labels, cutflow.nevonecut)))
        cumcuts = list(map(tuple, zip(cutflow.labels, cutflow.nevcutflow)))
        nmocuts = list(map(tuple, zip(nmo.labels, nmo.nev)))
        ret = SelectionFlow(cutflow=cumcuts, one_cut=onecut, n_minus_one=nmocuts)
        if self.parent_selection is not None:
            parent_cutflow = self.parent_selection.getSelectionFlow()
            ret = parent_cutflow.concatChild(ret)
        return ret


@dataclass
class Selection:
    select_from: SelectionSet
    names: tuple[str] = field(default_factory=tuple)

    def __add__(self, name):
        self.names = self.names + (name,)
        return self

    def getSelectionFlow(self):
        return self.select_from.getSelectionFlow(self.names)

    def getMask(self):
        return self.select_from.getMask(self.names)

    def __eq__(self, other):
        return self.select_from == other.select_from and self.names == other.names

class Selector:
    def __init__(self, selection, selection_set):
        self.selection = selection
        self.selection_set = selection_set

    def add(self, name, mask):
        self.selection += name
        return self.selection_set.addMask(name, mask)
