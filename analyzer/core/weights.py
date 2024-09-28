from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
from coffea.analysis_tools import Weights

from .sector import SubSectorId


@dataclass
class WeightManager:
    weights: defaultdict[SubSectorId, Weights] = field(
        default_factory=lambda: defaultdict(lambda: Weights(None, storeIndividual=True))
    )

    __cache: dict[(SubSectorId, str), Any] = field(default_factory=dict)

    def add(self, subsector_id, weight_name, central, variations=None):
        if subsector_id not in self.weights:
            if isinstance(central, ak.Array):
                self.weights[subsector_id] = Weights(ak.num(central, axis=0), storeIndividual=True)
                
        if variations:
            systs = [(x, *y) for x, y in variations.items()]
            name, up, down = list(map(list, zip(*systs)))
            self.weights[subsector_id].add_multivariation(
                weight_name, central, name, up, down
            )
        else:
            self.weights[subsector_id].add(weight_name, central)



    def variations(self, subsector_id):
        return list(self.weights[subsector_id].variations)

    def weight_names(self, subsector_id):
        return list(self.weights[subsector_id]._weights)

    def totalWeight(self, subsector_id):
        return (
            ak.sum(self.weight(subsector_id), axis=0),
            ak.sum(self.weight(subsector_id) ** 2, axis=0),
        )

    def weight(self, subsector_id, modifier=None, include=None, exclude=None):
        inc = include or []
        exc = exclude or []
        k = (subsector_id, modifier, tuple(inc), tuple(exc))

        if k in self.__cache:
            return self.__cache[k]

        weights = self.weights[subsector_id]
        if include or exclude:
            ret = weights.partial_weight(modifier=modifier, include=inc, exclude=exc)
        else:
            ret = weights.weight(modifier)

        self.__cache[k] = ret

        return ret

    def getSubSectorWeighter(self, subsector_id):
        return WeightManager.SubSectorWeighter(self, subsector_id)

    class SubSectorWeighter:
        def __init__(self, parent, subsector_id):
            self.parent = parent
            self.subsector_id = subsector_id

        def weight(self, *args, **kwargs):
            return self.parent.weight(self.subsector_id, *args, **kwargs)

        def add(self, *args, **kwargs):
            return self.parent.add(self.subsector_id, *args, **kwargs)

        @property
        def variations(self):
            return self.parent.variations(self.subsector_id)

        @property
        def weight_names(self):
            return self.parent.weight_names(self.subsector_id)
