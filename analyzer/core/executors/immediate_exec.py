from __future__ import annotations

import copy
from typing import Literal


import analyzer.core.results as core_results
from analyzer.utils.file_tools import extractCmsLocation
from analyzer.utils.structure_tools import accumulate
from .executor import Executor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from .preprocessing_tools import preprocess
import logging

logger = logging.getLogger(__name__)


class ImmediateExecutor(Executor):
    executor_type: Literal["immediate"] = "immediate"
    catch_exceptions: bool = False
    step_size: int = 100000

    def __run_task(self, task):
        fs = preprocess(task, test_mode=self.test_mode)
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
        for task in tasks:
            print(f"Running {task.sample_id}")
            task.file_set.step_size = task.file_set.step_size or self.step_size
            logger.info(f"Running task {task.sample_id}")
            result, fs, processed = self.__run_task(task)
            if result is not None:
                r = core_results.SampleResult(
                    sample_id=task.sample_id,
                    file_set_ran=fs,
                    file_set_processed=processed,
                    params=task.sample_params,
                    results=result,
                )
                if result_complete_callback is not None:
                    result_complete_callback(task.sample_id, r)
                else:
                    final_result[task.sample_id] = r

        return final_results
