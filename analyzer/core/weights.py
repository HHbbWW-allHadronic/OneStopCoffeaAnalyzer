from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
from coffea.analysis_tools import Weights

from .sector import SubSectorId
from pydantic import BaseClass, Field


class Weighter:
    def __init__(self, size=None, ignore_systematics=False):
        self.weights = Weights(size, storeIndividual=True)
        self.ignore_systematics = ignore_systematics
        self.__cache = {}

    def add(self, weight_name, central, variations=None):
        if variations and not self.ignore_systematics:
            systs = [(x, *y) for x, y in variations.items()]
            name, up, down = list(map(list, zip(*systs)))
            self.weights.add_multivariation(weight_name, central, name, up, down)
        else:
            self.weights.add(weight_name, central)

    @property
    def variations(self):
        return list(self.weights.variations)

    @property
    def weight_names(self):
        return list(self.weights._weights)

    # @property
    # def total_weight(self):
    #     return (
    #         ak.sum(self.weight(), axis=0),
    #         ak.sum(self.weight() ** 2, axis=0),
    #     )

    def weight(self, modifier=None, include=None, exclude=None):
        inc = include or []
        exc = exclude or []
        k = (modifier, tuple(inc), tuple(exc))
        if k in self.__cache:
            return self.__cache[k]
        if include or exclude:
            ret = self.weights.partial_weight(
                modifier=modifier, include=inc, exclude=exc
            )
        else:
            ret = self.weights.weight(modifier)
        self.__cache[k] = ret
        return ret
