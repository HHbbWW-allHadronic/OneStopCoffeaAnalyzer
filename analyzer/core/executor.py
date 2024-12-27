from rich import print

import dask
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from pydantic import BaseModel

import analyzer.core.results as results
import analyzer.core.analyzer as ac
from dataclasses import dataclass
from analyzer.datasets import FileSet, SampleId, SampleParams


@dataclass
class ExecutionUnit:
    sample_id: SampleId
    sample_params: SampleParams
    file_set: FileSet
    analyzer: ac.Analyzer


class CondorExecutor(BaseModel):
    pass


class DaskExecutor(BaseModel):
    def run(self, units):
        ret = {}
        all_events = {}
        for unit in units:
            cds = unit.file_set.toCoffeaDataset()
            print(cds)
            events, report = NanoEventsFactory.from_root(
                cds["files"],
                schemaclass=NanoAODSchema,
                uproot_options=dict(
                    allow_read_errors_with_report=True,
                    timeout=30,
                ),
                known_base_form=cds["form"]
            ).events()
            all_events[unit.sample_id] = (events, report)
        for unit in units:
            ret[unit.sample_id] = unit.analyzer.run(
                all_events[unit.sample_id][0], unit.sample_params
            )
        print(ret)
        # to_compute = {x: y.model_dump() for x, y in ret.items()}
        # ret = dask.compute(to_compute)[0]
        # ret = {x: results.SubSectorResult(**y) for x, y in ret.items()}
        # return ret


class ImmediateExecutor:
    pass
