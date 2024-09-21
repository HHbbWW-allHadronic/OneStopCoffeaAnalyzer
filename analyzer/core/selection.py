import functools as ft
import logging
import operator as op
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from analyzer.datasets import SampleId
from analyzer.utils.structure_tools import accumulate
from coffea.analysis_tools import PackedSelection
from pydantic import BaseModel, ConfigDict

from .common_types import Scalar
from .configuration import AnalysisStage
from .sector import SectorId

logger = logging.getLogger(__name__)


@dataclass
class Selection:
    or_names: tuple[str] = field(default_factory=tuple)
    and_names: tuple[str] = field(default_factory=tuple)

    def addOne(self, name, type="and"):
        if type == "and":
            s = Selection(or_names=tuple(), and_names=(name,))
        else:
            s = Selection(or_names=(name,), and_names=tuple())
        return self.merge(s)

    def merge(self, other):
        def addTuples(t, o):
            to_add = [x for x in o if x not in t]
            return tuple([*t, *to_add])

        return Selection(
            addTuples(self.or_names, other.or_names),
            addTuples(self.and_names, other.and_names),
        )


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


@dataclass
class SampleSelection:
    """
    Selection for a single sample.
    Stores the preselection and selection masks.
    The selection mask is relative to the "or" of all the preselection cuts.

    The general flow for a single sample looks like


    AllEvents -> Generate Preselection Masks -> AllEvents[PassAnyPreselectionCut]
    -> Generate Selection Mask -> Add Preselection Masks to Selection Mask
    -> Apply Appropriate Cuts for Each Region -> Final Trimmed Events

    """

    preselection_mask: PackedSelection = field(default_factory=PackedSelection)
    selection_mask: PackedSelection = field(default_factory=PackedSelection)

    def getPreselectionMask(self):
        names = self.preselection_mask.names
        if not names:
            return None
        return self.preselection_mask.any(*names)

    def addPreselectionMaskToSelection(self, complete_mask):
        """After finalizing selection  we want to add the preselection cuts to the analysis selection, so that we can generate cutflows etc.
        This function takes each preselection mask and adds the appropriate portion to the final selection, based on complete_mask, which should be the mask used to the compute the events post-preselection (ie the events that are ultimately selected on)
        """
        for n in self.preselection_mask.names:
            if complete_mask is None:
                self.addMask(
                    n, self.preselection_mask.any(n), stage=AnalysisStage.Selection
                )
            else:
                self.addMask(
                    n,
                    self.preselection_mask.any(n)[complete_mask],
                    stage=AnalysisStage.Selection,
                )

    def addMask(self, name, mask, type="and", stage=AnalysisStage.Preselection):
        if stage == AnalysisStage.Preselection:
            target = self.preselection_mask
        else:
            target = self.selection_mask

        names = target.names
        if not name in names:
            target.add(name, mask)

    def getMask(self, or_names, and_names, stage=AnalysisStage.Preselection):
        if not (or_names or and_names):
            return None
        if stage == AnalysisStage.Preselection:
            sel = self.preselection_mask
        else:
            sel = self.selection_mask

        names = sel.names
        t = sel.any(*or_names) if or_names else None
        a = sel.all(*and_names) if and_names else None
        return ft.reduce(op.and_, (x for x in [t, a] if x is not None))

    def getCutflow(self, names, stage=AnalysisStage.Selection):
        if stage == AnalysisStage.Preselection:
            sel = self.preselection_mask
        else:
            sel = self.selection_mask

        nmo = sel.nminusone(*names).result()
        cutflow = sel.cutflow(*names).result()
        onecut = list(map(tuple, zip(cutflow.labels, cutflow.nevonecut)))
        cumcuts = list(map(tuple, zip(cutflow.labels, cutflow.nevcutflow)))
        nmocuts = list(map(tuple, zip(nmo.labels, nmo.nev)))
        return Cutflow(cutflow=cumcuts, one_cut=onecut, n_minus_one=nmocuts)


@dataclass
class SelectionManager:
    selection_masks: defaultdict[SampleId, SampleSelection] = field(
        default_factory=lambda: defaultdict(SampleSelection)
    )

    selections: defaultdict[SectorId, Selection] = field(
        default_factory=lambda: defaultdict(Selection)
    )

    __computed_preselections: dict[str, Any] = field(default_factory=dict)

    def register(
        self, sector_id, name, mask, type="and", stage=AnalysisStage.Preselection
    ):
        logger.debug(
            f'Adding selection cut "{name}" of type "{type}" for stage "{stage}" for {sector_id}'
        )
        self.selections[sector_id] = self.selections[sector_id].addOne(name, type=type)
        self.selection_masks[sector_id.sample_id].addMask(name, mask, stage=stage)

    def maskPreselection(self, sample_id, events):
        logger.debug(f'Applying preselection mask  for "{sample_id}"')
        mask = self.selection_masks[sample_id].getPreselectionMask()
        self.__computed_preselections[sample_id] = mask
        if mask is None:
            return events
        else:
            return events[mask]

    def addPreselectionMasks(self):
        for sample_id, sel in self.selection_masks.items():
            logger.debug(
                f'Adding preselection mask to selection mask for "{sample_id}"'
            )
            m = self.__computed_preselections[sample_id]
            sel.addPreselectionMaskToSelection(m)

    def maskSector(self, sector_id, events):
        logger.debug(f'Applying selection mask  for "{sector_id}"')
        s = self.selections[sector_id]
        sample_id = sector_id.sample_id
        mask = self.selection_masks[sample_id].getMask(
            s.or_names, s.and_names, stage=AnalysisStage.Selection
        )
        if mask is None:
            return events
        else:
            return events[mask]

    def getCutflow(self, sector_id):
        sid = sector_id.sample_id
        sel = self.selections[sector_id]
        all_names = sel.or_names + sel.and_names
        print(all_names)
        return self.selection_masks[sid].getCutflow(all_names)
