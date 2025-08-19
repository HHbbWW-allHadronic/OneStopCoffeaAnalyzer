from __future__ import annotations

import copy
from typing import Literal


import analyzer.core.results as core_results
from analyzer.utils.file_tools import extractCmsLocation
from analyzer.utils.structure_tools import accumulate
from .executor import Executor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory



class ImmediateExecutor(Executor):
    executor_type: Literal["immediate"] = "immediate"
    catch_exceptions: bool = True
    step_size: int = 100000

    def _preprocess(self, tasks):
        return preprocess(
            tasks,
            default_step_size=self.step_size,
            scheduler="single-threaded",
            test_mode=self.test_mode,
        )

    def __run_task(self, k, task):
        fs = self._preprocess({k: task})[k]
        processed = copy.deepcopy(fs)
        cds = fs.toCoffeaDataset()
        ret = None
        for fname, data in cds["files"].items():
            for i, (start, end) in enumerate(data["steps"]):
                try:
                    events = NanoEventsFactory.from_root(
                        {fname: data["object_path"]},
                        schemaclass=NanoAODSchema,
                        delayed=False,
                        entry_start=start,
                        entry_stop=end,
                    ).events()
                    if ret is None:
                        ret = task.analyzer.run(events, task.sample_params)
                    else:
                        ret = accumulate(
                            [ret, task.analyzer.run(events, task.sample_params)]
                        )
                except Exception as e:
                    logger.warn(
                        f"An exception occurred while running {fname} ({start},{end}).\n"
                        f"{e}"
                    )
                    if not self.catch_exceptions:
                        raise
                    else:
                        processed.dropChunk(extractCmsLocation(fname), [start, end])
        return ret, fs, processed

    def run(self, tasks, result_complete_callback=None):

        logger.info(f"Starting run with immediate executor")

        final_results = {}
        for k, task in tasks.items():
            try:
                logger.info(f"Running task {k}")
                result, fs, processed = self.__run_task(k, task)
                if result is not None:
                    r = core_results.SampleResult(
                        sample_id=tasks[k].sample_id,
                        file_set_ran=fs,
                        file_set_processed=processed,
                        params=tasks[k].sample_params,
                        results=result,
                    )
                    if result_complete_callback is not None:
                        result_complete_callback(k, r)
                    else:
                        final_result[k] = r
            except Exception as e:
                logger.warn(f"An exception occurred while running {k}.\n {e}")
                if not self.catch_exceptions:
                    raise

        return final_results
