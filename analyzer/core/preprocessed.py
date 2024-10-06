import copy
import itertools as it
import logging
from collections import namedtuple
from typing import Any, Optional

import analyzer.utils.structure_tools as utils

# import analyzer.core.preprocess as dst
import coffea.dataset_tools as dst
import distributed
import pydantic as pyd
from analyzer.datasets import SampleId
from analyzer.utils.file_tools import extractCmsLocation, stripPort
from coffea.dataset_tools.preprocess import DatasetSpec

logger = logging.getLogger(__name__)

Chunk = namedtuple("Chunk", "file start end")


class SamplePreprocessed(pyd.BaseModel):
    """A preprocessed samples, containing information about the step breakdown of the files in sample_id."""

    sample_id: SampleId
    chunk_info: dict[str, dict[str, Any]]
    step_size: int
    form: Optional[str] = None
    limit_chunks: Optional[set[Chunk]] = None
    file_retrieval_kwargs: Optional[dict[str, Any]] = None

    def getCoffeaDataset(
        self, dataset_repo, allow_incomplete=True, **kwargs
    ) -> DatasetSpec:
        sample = dataset_repo.getSample(self.sample_id)
        fdict = sample.fdict
        if len(self.chunk_info) != len(fdict) and not allow_incomplete:
            raise RuntimeError(f"Preprocessed dataset is not complete.")

        coffea_dataset = {
            "files": {f: copy.deepcopy(data) for f, data in self.chunk_info.items()},
            "form": self.form,
        }

        if self.limit_chunks:
            d = {}
            for fname, *step in self.limit_chunks:
                if fname in d:
                    d[fname].append(step)
                else:
                    d[fname] = [step]

            files = {k: v for k, v in coffea_dataset["files"].items() if k in d}
            for f, steps in d.items():
                files[f]["steps"] = steps
            coffea_dataset["files"] = files

        coffea_dataset["files"] = {
            fdict[f].getFile(**kwargs): data
            for f, data in coffea_dataset["files"].items()
        }
        return coffea_dataset

    def addCoffeaChunks(self, other):

        # if self.dataset_input != other.dataset_input:
        #    raise ValueError("Cannot add Preprocessed Datasets with differing inputs")
        new_data = copy.deepcopy(self.chunk_info)

        updates = {
            extractCmsLocation(fname): data for fname, data in other["files"].items()
        }
        new_data.update(updates)
        return SamplePreprocessed(
            sample_id=self.sample_id,
            chunk_info=new_data,
            step_size=self.step_size,
            form=self.form,
            limit_chunks=self.limit_chunks,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def missingFiles(self, dataset_repo):
        """Get files that were not successfully preprocessed"""

        a = set(x.cmsLocation() for x in dataset_repo.getSample(self.sample_id).files)
        b = set(self.chunk_info)
        return list(a - b)

    def missingCoffeaDataset(self, dataset_repo, **kwargs):
        """Get a CoffeaDataset containing only those files that were not preprocessed.
        This is used to patch preprocessed files that failed to compute.

        """

        mf = self.missingFiles(dataset_repo)
        sample = dataset_repo.getSample(self.sample_id)
        fdict = sample.fdict
        return {"files": {fdict[x].getFile(**kwargs): fdict[x].object_path for x in mf}}

    @property
    def chunks(self):
        if self.limit_chunks:
            return self.limit_chunks
        return set(
            it.chain(
                (fname, *s)
                for fname, vals in self.chunk_info.items()
                for s in vals["steps"]
            )
        )


def getCoffeaDatasetForSample(dataset_repo, sample_id, **kwargs):
    fdict = dataset_repo.getSample(sample_id).fdict
    return {
        str(sample_id): {
            "files": {x.getFile(**kwargs): x.object_path for x in fdict.values()}
        }
    }


def preprocessBulk(
    dataset_repo, samples, file_retrieval_kwargs=None, step_size=None, **kwargs
):
    logger.info(f"Preprocessing {len(samples)} samples.")
    if file_retrieval_kwargs is None:
        file_retrieval_kwargs = {}

    logger.debug(f"Preprocessing with file args: {file_retrieval_kwargs}.")
    mapping = {str(x): x for x in samples}
    all_inputs = utils.accumulate(
        [getCoffeaDatasetForSample(dataset_repo, x, **file_retrieval_kwargs) for x in samples]
    )
    logger.debug(f"Launching preprocessor.")
    logger.info(distributed.client._get_global_client())
    out, bad = dst.preprocess(
        all_inputs,
        save_form=True,
        skip_bad_files=True,
        uproot_options={"timeout": 30},
        step_size=step_size,
        **kwargs,
    )

    def backToSampleFile(v):
        return {extractCmsLocation(fname): data for fname, data in v["files"].items()}

    ret = [
        SamplePreprocessed(
            sample_id=mapping[k],
            chunk_info=backToSampleFile(v),
            step_size=step_size,
            file_retrieval_kwargs=file_retrieval_kwargs,
        )
        for k, v in out.items()
    ]
    logger.debug(f"Preprocessed {len(ret)} samples")
    return ret


def preprocessRaw(inputs, **kwargs):
    out, bad = dst.preprocess(
        inputs,
        save_form=True,
        skip_bad_files=True,
        uproot_options={"timeout": 15},
        **kwargs,
    )
    return out
