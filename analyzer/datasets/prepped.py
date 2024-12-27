import copy
from collections import defaultdict
from typing import Any, Optional


import awkward as ak
from analyzer.configuration import CONFIG
from analyzer.utils.file_tools import extractCmsLocation
from pydantic import BaseModel
import analyzer.datasets.files as adf


if CONFIG.PRETTY_MODE:
    pass


# class SamplePreprocessed(pyd.BaseModel):
#     """A preprocessed samples, containing information about the step breakdown of the files in sample_id."""
#
#     chunk_info: DatasetSpec
#     step_size: int
#     form: Optional[str] = None
#     limit_chunks: Optional[set[Chunk]] = None
#     file_retrieval_kwargs: Optional[dict[str, Any]] = None
#
#     def getCoffeaDataset(
#         self, dataset_repo, allow_incomplete=True, **kwargs
#     ) -> DatasetSpec:
#         sample = dataset_repo.getSample(self.sample_id)
#         fdict = sample.fdict
#         if len(self.chunk_info) != len(fdict) and not allow_incomplete:
#             raise RuntimeError(f"Preprocessed dataset is not complete.")
#
#         coffea_dataset = {
#             "files": {f: copy.deepcopy(data) for f, data in self.chunk_info.items()},
#             "form": self.form,
#         }
#
#         if self.limit_chunks:
#             d = {}
#             for fname, *step in self.limit_chunks:
#                 if fname in d:
#                     d[fname].append(step)
#                 else:
#                     d[fname] = [step]
#
#             files = {k: v for k, v in coffea_dataset["files"].items() if k in d}
#             for f, steps in d.items():
#                 files[f]["steps"] = steps
#                 coffea_dataset["files"] = files
#
#         coffea_dataset["files"] = {
#             fdict[f].getFile(**kwargs): data
#             for f, data in coffea_dataset["files"].items()
#         }
#         return coffea_dataset
#
#     def addCoffeaChunks(self, other):
#
#         # if self.dataset_input != other.dataset_input:
#         #    raise ValueError("Cannot add Preprocessed Datasets with differing inputs")
#         new_data = copy.deepcopy(self.chunk_info)
#
#         updates = {
#             extractCmsLocation(fname): data for fname, data in other["files"].items()
#         }
#         new_data.update(updates)
#         return SamplePreprocessed(
#             sample_id=self.sample_id,
#             chunk_info=new_data,
#             step_size=self.step_size,
#             form=self.form,
#             limit_chunks=self.limit_chunks,
#             file_retrieval_kwargs=self.file_retrieval_kwargs,
#         )
#
#     def missingFiles(self, dataset_repo):
#         """Get files that were not successfully preprocessed"""
#
#         a = set(x.cmsLocation() for x in dataset_repo.getSample(self.sample_id).files)
#         b = set(self.chunk_info)
#         return list(a - b)
#
#     def missingCoffeaDataset(self, dataset_repo, **kwargs):
#         """Get a CoffeaDataset containing only those files that were not preprocessed.
#         This is used to patch preprocessed files that failed to compute.
#
#         """
#
#         mf = self.missingFiles(dataset_repo)
#         sample = dataset_repo.getSample(self.sample_id)
#         fdict = sample.fdict
#         return {"files": {fdict[x].getFile(**kwargs): fdict[x].object_path for x in mf}}
#
#     @property
#     def chunks(self):
#         if self.limit_chunks:
#             return self.limit_chunks
#         return set(
#             it.chain(
#                 (fname, *s)
#                 for fname, vals in self.chunk_info.items()
#                 for s in vals["steps"]
#             )
#         )


class FileSet(BaseModel):
    files: dict[str, tuple[adf.SampleFile, dict]]
    step_size: Optional[int]
    form: Optional[str] = None
    file_retrieval_kwargs: Optional[dict[str, Any]] = None

    def intersect(self, other):
        ret = copy.deepcopy(self.files)
        for fname, (_, data) in ret.items():
            if not (data["steps"] is None and other[fname]["steps"] is None):
                data["steps"] = [
                    s for s in data["steps"] if s in other["fname"]["steps"]
                ]
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def add(self, other):
        ret = copy.deepcopy(self.files)
        for fname, (_, data) in ret.items():
            if not (data["steps"] is None and other[fname]["steps"] is None):
                data["steps"] = (data["steps"] or []) + (other.get(steps) or [])
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def sub(self, other):
        ret = copy.deepcopy(self.files)
        for fname, (_, data) in ret.items():
            if not (data["steps"] is None and other[fname]["steps"] is None):
                data["steps"] = [
                    s for s in data["steps"] if s not in other["fname"]["steps"]
                ]
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    @property
    def empty(self):
        return not files or all(not x["steps"] for _, (_, x) in files.items())

    def getSampleFile(self, identity):
        return next(x for x, _ in self.files if x == identity)

    def justFailed(self, report):
        failures = report[~ak.is_none(report.exception)]

        failed = defaultdict(list)
        ret = copy.deepcopy(self.files)

        for failure in failures:
            args_as_types = tuple(eval(arg) for arg in failure.args)
            fname, object_path, start, stop, is_step = args_as_types
            failed[extractCmsLocation(fname)].append([start, stop])

        for fname in list(ret.keys()):
            if fname in failures:
                ret[fname][1]["steps"] = failures[fname]
            else:
                del ret[fname]

        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=form,
        )

    def justProcessed(self, report):
        passed_report = report[ak.is_none(report.exception)]
        passed = defaultdict(list)
        ret = copy.deepcopy(self.files)

        for ok in passed_report:
            args_as_types = tuple(eval(arg) for arg in ok.args)
            fname, object_path, start, stop, is_step = args_as_types
            passed[extractCmsLocation(fname)].append([start, stop])

        for fname in list(ret.keys()):
            if fname in passed:
                ret[fname][1]["steps"] = passed[fname]
            else:
                del ret[fname]

        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=form,
        )


    def updateFromCoffea(self, coffea_fileset):
        ret = copy.deepcopy(self.files)
        for fname, data in coffea_fileset["files"].items():
            n = extractCmsLocation(fname)
            ret[n][1]["steps"] = data["steps"]
            ret[n][1]["uuid"] = data["uuid"]
            ret[n][1]["num_entries"] = data["num_entries"]
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=coffea_fileset["form"],
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )


    def toCoffeaDataset(self):
        coffea_dataset = {
            "files": {
                f.getFile(**self.file_retrieval_kwargs): copy.deepcopy(data)
                for f, data in self.files.values()
            },
            "form": self.form,
        }
        return coffea_dataset

