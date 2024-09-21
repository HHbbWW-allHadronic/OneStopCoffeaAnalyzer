from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
from coffea.analysis_tools import Weights

from .sector import SectorId


@dataclass
class WeightManager:
    weights: defaultdict[SectorId, Weights] = field(
        default_factory=lambda: defaultdict(lambda: Weights(None, storeIndividual=True))
    )

    __cache: dict[(SectorId, str), Any] = field(default_factory=dict)

    def add(self, sector_id, weight_name, central, variations=None):
        if variations:
            systs = [(x, *y) for x, y in variations.items()]
            name, up, down = list(map(list, zip(*systs)))
            self.weights[sector_id].add_multivariation(
                weight_name, central, name, up, down
            )
        else:
            self.weights[sector_id].add(weight_name, central)



    def variations(self, sector_id):
        return list(self.weights[sector_id].variations)

    def weight_names(self, sector_id):
        return list(self.weights[sector_id]._weights)

    def totalWeight(self, sector_id):
        return (
            ak.sum(self.weight(sector_id), axis=0),
            ak.sum(self.weight(sector_id) ** 2, axis=0),
        )

    def weight(self, sector_id, modifier=None, include=None, exclude=None):
        inc = include or []
        exc = exclude or []
        k = (sector_id, modifier, tuple(inc), tuple(exc))

        if k in self.__cache:
            return self.__cache[k]

        weights = self.weights[sector_id]
        if include or exclude:
            ret = weights.partial_weight(modifier=modifier, include=inc, exclude=exc)
        else:
            ret = weights.weight(modifier)

        self.__cache[k] = ret

        return ret

    def getSectorWeighter(self, sector_id):
        return WeightManager.SectorWeighter(self, sector_id)

    class SectorWeighter:
        def __init__(self, parent, sector_id):
            self.parent = parent
            self.sector_id = sector_id

        def weight(self, *args, **kwargs):
            return self.parent.weight(self.sector_id, *args, **kwargs)

        def add(self, *args, **kwargs):
            return self.parent.add(self.sector_id, *args, **kwargs)

        @property
        def variations(self):
            return self.parent.variations(self.sector_id)

        @property
        def weight_names(self):
            return self.parent.weight_names(self.sector_id)
