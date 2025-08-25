from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field, AliasChoices
import dataclasses
from analyzer.configuration import CONFIG
import logging

logger = logging.getLogger(__name__)


class JetCorrectionInfo(BaseModel):
    files: dict[str, str]
    jec: dict[str, str]
    jer: dict[str, str]
    jet_names: dict[str,str]


class Era(BaseModel):
    name: str = Field(alias=AliasChoices("era_name", "name"))
    energy: float
    lumi: float
    golden_json: str

    trigger_names: dict[str, str | None] = Field(default_factory=dict)
    trigger_plateaus: dict[str, float | dict[str, float]] = Field(default_factory=dict)
    sections: dict[str, dict[str, Any]] = Field(default_factory=dict)

    pileup_scale_factors: dict[str, str]
    btag_scale_factors: dict[str, Any]
    jet_pileup_id: dict[str, Any] | None = None
    jet_veto_maps: dict[str, Any] | None = None
    jet_corrections: JetCorrectionInfo | None = None

    @property
    def params(self):
        return self.model_dump(by_alias=True)


@dataclasses.dataclass
class EraRepo:
    eras: dict[str, Era] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key):
        return self.eras[key]

    def __iter__(self):
        return iter(self.eras)

    def load(self, directory):
        logger.info(f"Loading eras from {directory}")
        directory = Path(directory)
        files = list(directory.rglob("*.yaml"))
        for f in files:
            with open(f, "r") as fo:
                logger.debug(f"Loading era from  file {f}")
                data = yaml.safe_load(fo)
                for d in data:
                    s = Era(**d)
                    if s.name in self.eras:
                        raise KeyError(
                            f"Dataset name '{s.name}' is already use. Please use a different name for this era."
                        )
                    self.eras[s.name] = s

    @staticmethod
    def getConfig():
        paths = CONFIG.ERA_PATHS
        repo = EraRepo()
        for path in paths:
            repo.load(path)
        return repo


if __name__ == "__main__":
    em = EraRepo()
    em.load("analyzer_resources/eras")
    print(em)
