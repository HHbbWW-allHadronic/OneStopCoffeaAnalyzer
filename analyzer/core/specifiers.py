import logging
from fnmatch import fnmatch
from typing import Optional, Union

from analyzer.datasets import SampleType
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SampleSpec(BaseModel):
    sample_name: Optional[Union[list[str], str]] = None
    era: Optional[Union[list[str], str]] = None
    sample_type: Optional[SampleType] = None

    def passes(self, sample, return_specificity=False):
        passes_names = not self.sample_name or any(
            fnmatch(sample.name, x) for x in self.sample_name
        )

        passes_era = not self.era or any(fntmatch(sample.era, x) for x in self.era)
        passes_type = not self.sample_type or (sample.sample_type == self.sample_type)
        passes = passes_names and passes_era and passes_type
        if return_specificity:
            if not passes:
                return 0
            exact_name = any(sample.name == x for x in self.sample_name)
            exact_era = any(sample.era == x for x in self.sample_era)
            return (
                100 * exact_name
                + 50 * (not exact_name and passes_names)
                + 30 * exact_era
                + 2 * (not exact_era and passes_era)
                + 5 * passes_type
            )

        else:
            return passes


class SectorSpec(BaseModel):
    sample_spec: Optional[SampleSpec] = None
    region_names: Optional[Union[list[str], str]] = None

    def passes(self, sector_id, return_specificity=False):
        passes_sample = not self.sample_spec or sample_spec.passes(
            sector.sample_id, sample_manager, return_specificity
        )
        passes_region = not self.region_names or any(
            fnmatch(sector.region_name, x) for x in self.region_names
        )
        passes = bool(passes_sample) and passes_region
        if return_specificity:
            exact_region = any(sector.region_name == x for x in self.region_names)
            return (
                80 * exact_region
                + 50 * (not exact_region and passes_region)
                + passes_sample
            )

        else:
            return passes
