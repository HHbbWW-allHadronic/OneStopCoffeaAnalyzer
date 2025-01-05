import copy
import itertools as it
from collections import defaultdict
from typing import Any
from rich import print


import awkward as ak
from analyzer.utils.file_tools import extractCmsLocation
from pydantic import BaseModel
import analyzer.datasets.files as adf


class FileSet(BaseModel):
    files: dict[str, tuple[adf.SampleFile, dict]]
    step_size: int | None
    form: str | None = None
    file_retrieval_kwargs: dict[str, Any] | None = None

    def intersect(self, other):
        common_files = set(self.files).intersection(other.files)
        ret = {}
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
        ret = {}
        for fname, (s, data) in it.chain(self.files.items(), other.files.items()):
            if fname not in ret:
                ret[fname] = (s, data)
            else:
                c = ret[fname][1]
                steps_init = c["steps"]
                steps_other = data["steps"]
                if steps_init is None and steps_other is None:
                    c["steps"] = None
                elif steps_init is None:
                    c["steps"] = steps_other
                elif steps_other is None:
                    c["steps"] = steps_init
                else:
                    steps_new = [x for x in steps_other if x not in steps_init]
                    c["steps"] = list(sorted(steps_init + steps_new))

        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def withoutFiles(self, other):
        ret = {x: y for x, y in self.files.items() if x not in other.files}
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def sub(self, other):
        ret = copy.deepcopy(self.files)
        common_files = set(self.files).intersection(other.files)

        for fname in common_files:
            ret_data = ret[fname][1]
            other_data = other.files[fname][1]

            new_steps = [
                x
                for x in (ret_data["steps"] or [])
                if x not in (other_data["steps"] or [])
            ]
            if not new_steps:
                del ret[fname]
            else:
                ret[fname][1]["steps"] = new_steps

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
        ret = not self.files or all(
            x["steps"] is not None and not x["steps"] for _, x in self.files.values()
        )
        return ret

    @property
    def events(self):
        return sum(
            s[1] - s[0] if s is not None else 0
            for x in self.files.values()
            for s in x[1]["steps"]
        )

    def dropChunk(self, fname, chunk):
        self.files[fname][1]["steps"].remove(chunk)

    def getSampleFile(self, identity):
        return next(x for x, _ in self.files if x == identity)

    def justChunked(self):
        ret = copy.deepcopy(self.files)
        for k in list(ret.keys()):
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
        for k in list(ret.keys()):
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

    def splitFiles(self, files_per_set):
        lst = list(self.files.items())
        files_split = {(i, i + n): dict(lst[i : i + n]) for i in range(0, len(lst), n)}
        return {
            k: FileSet(
                files=v,
                step_size=self.step_size,
                form=coffea_fileset["form"],
                file_retrieval_kwargs=self.file_retrieval_kwargs,
            )
            for k, v in files_split.items()
        }

    def toCoffeaDataset(self, simple=False):
        if simple:
            coffea_dataset = {
                "files": {
                    f.getFile(**self.file_retrieval_kwargs): data["object_path"]
                    for f, data in self.files.values()
                },
                "form": self.form,
            }
        else:
            coffea_dataset = {
                "files": {
                    f.getFile(**self.file_retrieval_kwargs): copy.deepcopy(data)
                    for f, data in self.files.values()
                },
                "form": self.form,
            }
        return coffea_dataset

    def asEmpty(self):
        return FileSet(
            files={},
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )

    def slice(self, files=None, chunks=None):
        ret = list(copy.deepcopy(self.files).items())
        if files is not None:
            ret = ret[files]
        ret = dict(ret)
        if chunks is not None:
            for _, x in ret.values():
                if x["steps"] is not None:
                    x["steps"] = x["steps"][chunks]
        return FileSet(
            files=ret,
            step_size=self.step_size,
            form=self.form,
            file_retrieval_kwargs=self.file_retrieval_kwargs,
        )


# def getPatch(target, processed):
