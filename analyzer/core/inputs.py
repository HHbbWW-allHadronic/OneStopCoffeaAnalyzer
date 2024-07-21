import copy
import logging
from dataclasses import dataclass
import itertools as it
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import analyzer.utils as utils
#import coffea.dataset_tools as dst
import analyzer.core.preprocess as dst
from analyzer.file_utils import stripPort, extractCmsLocation
from coffea.dataset_tools.preprocess import DatasetSpec

logger = logging.getLogger(__name__)


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


@dataclass
class AnalyzerInput:
    dataset_name: str
    fill_name: str
    files: Dict[str, Any]
    profile: Any
    lumi_json: Optional[str] = None

    def getCoffeaDataset(self, **kwargs):
        return {
            self.dataset_name: {
                "files": {
                    x.getFile(**kwargs): x.object_path for x in self.files.values()
                }
            }
        }


@dataclass
class DatasetPreprocessed:
    dataset_input: AnalyzerInput
    chunk_info: Dict[str, Dict[str, Any]]
    form: str = None
    limit_chunks: Set[Tuple[str, int, int]] = None

    @staticmethod
    def fromDatasetInput(dataset_input, **kwargs):
        out, x = dst.preprocess(
            dataset_input.getCoffeaDataset(**kwargs), save_form=True, **kwargs
        )
        return DatasetPreprocessed(dataset_input, out[dataset_input.dataset_name])

    def getCoffeaDataset(self, allow_incomplete=True, **kwargs) -> DatasetSpec:
        if (
            len(self.chunk_info) != len(self.dataset_input.files)
            and not allow_incomplete
        ):
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
            self.dataset_input.files[f].getFile(**kwargs): data
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

    def missingFiles(self):
        a = set(self.dataset_input.files)
        b = set(self.chunk_info)
        return list(a - b)

    def missingCoffeaDataset(self, **kwargs):
        mf = self.missingFiles()
        return {
            self.dataset_name: {
                "files": {
                    self.dataset_input.files[x]
                    .getFile(**kwargs): self.dataset_input.files[x]
                    .object_path
                    for x in mf
                }
            }
        }

    @property
    def dataset_name(self):
        return self.dataset_input.dataset_name

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


def preprocessBulk(
    dataset_input: Iterable[AnalyzerInput], file_retrieval_kwargs=None, **kwargs
):
    if file_retrieval_kwargs is None:
        file_retrieval_kwargs = {}
    mapping = {x.dataset_name: x for x in dataset_input}
    all_inputs = utils.accumulate(
        [x.getCoffeaDataset(**file_retrieval_kwargs) for x in dataset_input]
    )
    out, bad = dst.preprocess(
        all_inputs,
        save_form=True,
        skip_bad_files=True,
        uproot_options={"timeout": 30},
        **kwargs,
    )

    def backToSampleFile(file_set, v):
        return {extractCmsLocation(fname): data for fname, data in v["files"].items()}

    ret = [
        DatasetPreprocessed(
            mapping[k], backToSampleFile(mapping[k].files, v), v["form"]
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
