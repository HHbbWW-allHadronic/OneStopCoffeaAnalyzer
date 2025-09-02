import abc
from rich import print
from enum import Enum, auto
import logging
from pathlib import Path

import yaml

from pydantic import BaseModel
from analyzer.configuration import CONFIG

from .style import Style, StyleSet

logger = logging.getLogger(__name__)
StyleLike = Style | str


class PostProcessorType(Enum):
    Normal = auto()
    Accumulator = auto()


class BasePostprocessor(BaseModel, abc.ABC):
    name: str

    @abc.abstractmethod
    def getExe(self, results):
        pass

    def getNeededHistograms(self):
        return []

    def init(self):
        if hasattr(self, "style_set") and isinstance(self.style_set, str):
            print("Loading style set")
            config_path = Path(CONFIG.STYLE_PATH) / self.style_set
            with open(config_path, "r") as f:
                d = yaml.safe_load(f)
            self.style_set = StyleSet(**d)
