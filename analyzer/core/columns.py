from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
from coffea.analysis_tools import Weights

from .sector import SubSectorId






@dataclass
class Column:
    name: str
    nominal_value: Any
    shape_variations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnManager:
    columns: defaultdict[SubSectorId, dict[str, Column]] = field(
        default_factory=lambda x: defaultdict(dict)
    )

    def add(self, subsector_id, name, nominal_value, variations=None):
        self.columns[subsector_id][name] = Column(
            name=name, nominal_value=nominal_value, shape_variations=variations
        )

    def get(self, subsector_id, name, variation=None):
        col = self.columns[subsector_id][name]
        if variation is None:
            return col.nominal_value
        else:
            return col.shape_variations[variation]


@dataclass(frozen=True)
class ShapeSystematicId:
    column_name: str
    variation_name: str
