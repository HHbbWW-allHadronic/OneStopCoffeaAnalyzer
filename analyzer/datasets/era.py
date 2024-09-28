from pathlib import Path
from typing import Optional, Any
import yaml
from pydantic import BaseModel, Field
import dataclasses
from analyzer.configuration import CONFIG
import logging

logger = logging.getLogger(__name__)


class Era(BaseModel):
    name: str
    energy: float
    lumi: float
    golden_json: str

    trigger_names: dict[str,str] = Field(default_factory=dict)
    sections: dict[str,dict[str,Any]] = Field(default_factory=dict)

    pileup_scale_factors: dict[str, str]
    btag_scale_factors: dict[str, str]
    jet_pileup_id: Optional[dict[str, Any]] = None
    jet_veto_maps: Optional[dict[str, Any]] = None


    @property
    def params(self):
        return self.dict(exclude=["name"])
    
    


@dataclasses.dataclass
class EraRepo:
    eras: dict[str, Era] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key):
        return self.eras[key]

    def load(self, directory):
        logger.info(f"Loading eras from {directory}")
        directory = Path(directory)
        files = list(directory.glob("*.yaml"))
        file_contents = {}
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
