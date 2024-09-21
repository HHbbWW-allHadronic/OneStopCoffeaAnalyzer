import copy
import itertools as it
import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Optional

import analyzer.utils.structure_tools as utils

# import analyzer.core.preprocess as dst
import coffea.dataset_tools as dst
import pydantic as pyd
from analyzer.datasets import SampleId
from analyzer.utils.file_tools import extractCmsLocation, stripPort
from coffea.dataset_tools.preprocess import DatasetSpec

logger = logging.getLogger(__name__)

Chunk = namedtuple("Chunk", "file start end")


def getMissingDataset(analyzer_input, coffea_dataset):
    cds = analyzer_input.coffea_dataset[analyzer_input.dataset_name]
    files = [(stripPort(x), x) for x in cds["files"]]
    cof_files = set(stripPort(x) for x in coffea_dataset["files"].keys())
    diff = {x[1] for x in files if not any(x[0] in y for y in cof_files)}
    return {
        analyzer_input.dataset_name: {
            "files": {
                fname: info for fname, info in cds["files"].items() if fname in diff
            }
        }
    }


class SamplePreprocessed(pyd.BaseModel):
    sample_id: SampleId
    chunk_info: dict[str, dict[str, Any]]
    form: Optional[str] = None
    limit_chunks: Optional[set[Chunk]] = None

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
            extractCmsLocation(fname): data
            for fname, data in other[self.dataset_name]["files"].items()
        }

        new_data.update(updates)

        return DatasetPreprocessed(
            self.dataset_input, new_data, self.form, self.limit_chunks
        )

    def missingFiles(self, dataset_repo):
        a = set(dataset_repo.getSample(self.sample_id))
        b = set(self.chunk_info)
        return list(a - b)

    def missingCoffeaDataset(self, dataset_repo, **kwargs):
        mf = self.missingFiles()
        sample = dataset_repo.getSample(self.sample_id)
        fdict = sample.fdict
        return {
            self.dataset_name: {
                "files": {fdict[x].getFile(**kwargs): fdict[x].object_path for x in mf}
            }
        }

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


def getCoffeaDataset(dataset_repo, sample_id, **kwargs):
    fdict = dataset_repo.getSample(sample_id).fdict
    from rich import print

    return {
        str(sample_id): {
            "files": {x.getFile(**kwargs): x.object_path for x in fdict.values()}
        }
    }


def preprocessBulk(dataset_repo, samples, file_retrieval_kwargs=None, **kwargs):
    if file_retrieval_kwargs is None:
        file_retrieval_kwargs = {}

    mapping = {str(x): x for x in samples}
    all_inputs = utils.accumulate(
        [getCoffeaDataset(dataset_repo, x, **file_retrieval_kwargs) for x in samples]
    )
    out, bad = dst.preprocess(
        all_inputs,
        save_form=True,
        skip_bad_files=True,
        uproot_options={"timeout": 30},
        **kwargs,
    )

    def backToSampleFile(v):
        return {extractCmsLocation(fname): data for fname, data in v["files"].items()}

    ret = [
        SamplePreprocessed(
            sample_id=mapping[k], chunk_info=backToSampleFile(v), form=v["form"]
        )
        for k, v in out.items()
    ]
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
