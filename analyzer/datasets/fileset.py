import copy
from collections import defaultdict
from typing import Any, Optional


import awkward as ak
from analyzer.configuration import CONFIG
from analyzer.utils.file_tools import extractCmsLocation
from pydantic import BaseModel
import analyzer.datasets.files as adf


class FileSet(BaseModel):
    files: dict[str, tuple[adf.SampleFile, dict]]
    step_size: Optional[int]
    form: Optional[str] = None
    file_retrieval_kwargs: Optional[dict[str, Any]] = None

    def intersect(self, other):
        # ret = copy.deepcopy(self.files)
        common_files = set(self.files).intersection(other.files)
        ret = {}
        print(common_files)
        for fname in common_files:
            data_this = self.files[fname][1]
            sf_this = self.files[fname][0]
            data_other = other.files[fname][1]
            data = copy.deepcopy(data_this)
            if not (data_this["steps"] is None and data_other["steps"] is None):
                data["steps"] = [
                    s
                    for s in (data_this["steps"] or [])
                    if s in (data_other["steps"] or [])
                ]
            ret[fname] = (sf_this, copy.deepcopy(data))

        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def add(self, other):
        ret = copy.deepcopy(self.files)
        for fname, (_, data) in ret.items():
            if fname not in other.files:
                continue
            if not (data["steps"] is None and other.files[fname]["steps"] is None):
                data["steps"] = (data["steps"] or []) + (
                    other.files[fname][1]["steps"] or []
                )
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def sub(self, other):
        ret = copy.deepcopy(self.files)
        for fname, (_, data) in ret.items():
            if fname not in other.files:
                continue
            if not (data["steps"] is None and other.files[fname]["steps"] is None):
                data["steps"] = [
                    s
                    for s in (data["steps"] or [])
                    if s not in (other[fname]["steps"] or [])
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
        return not self.files or all(not x["steps"] for _, (_, x) in self.files.items())

    def dropChunk(self, fname ,chunk):
        self.files[fname][1]["steps"].remove(chunk)

    def getSampleFile(self, identity):
        return next(x for x, _ in self.files if x == identity)

    def justChunked(self):
        ret = copy.deepcopy(self.files)
        for k in ret.keys():
            if ret[k][1]["steps"] is None:
                del ret[k]
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def justUnchunked(self):
        ret = copy.deepcopy(self.files)
        for k in ret.keys():
            if ret[k][1]["steps"] is not None:
                del ret[k]
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def _getExceptionState(self, operator, report):
        failures = report[operator(report.exception)]

        failed = defaultdict(list)
        ret = copy.deepcopy(self.files)

        for failure in failures:
            args_as_types = tuple(eval(arg) for arg in failure.args)
            fname, object_path, start, stop, is_step = args_as_types
            failed[extractCmsLocation(fname)].append([start, stop])

        for fname in list(ret.keys()):
            if fname in failed:
                ret[fname][1]["steps"] = failed[fname]
            else:
                del ret[fname]

        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def justFailed(self, report):
        return self._getExceptionState(lambda e: ~ak.is_none(e), report)

    def justProcessed(self, report):
        return self._getExceptionState(lambda e: ak.is_none(e), report)

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

    def toCoffeaDataset(self, simple=False):
        if simple:
            coffea_dataset = {
            "files": {
                f.getFile(**self.file_retrieval_kwargs): data["object_path"]
                for f, data in self.files.values()
            },
            "form": self.form}
        else:
            coffea_dataset = {
            "files": {
                f.getFile(**self.file_retrieval_kwargs): copy.deepcopy(data)
                for f, data in self.files.values()
            },
            "form": self.form,
        }
        return coffea_dataset

# def getPatch(target, processed):

        


