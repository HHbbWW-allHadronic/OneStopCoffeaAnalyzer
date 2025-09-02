from __future__ import annotations

from pathlib import Path


import coffea.dataset_tools as dst
from analyzer.configuration import CONFIG
from analyzer.configuration import CONFIG


def getPrepCachePath(task):
    base = Path(CONFIG.PREPROCESS_CACHE_PATH)
    base / str(task.sample_id) / str()

    

def preprocess(task, test_mode=False):
    unchunked = task.file_set.justUnchunked()
    if test_mode:
        unchunked = unchunked.slice(files=slice(0,1))
    if not unchunked.empty:
        to_prep = {task.sample_id: unchunked.toCoffeaDataset()}
        out, all_items = dst.preprocess(
            to_prep,
            save_form=True,
            skip_bad_files=True,
            step_size=task.file_set.step_size,
            allow_empty_datasets=True,
        )
        if out:
            file_set_prepped = task.file_set.updateFromCoffea(
                out[task.sample_id]
            ).justChunked()
        else:
            file_set_prepped = task.file_set.justChunked()
    else:
        file_set_prepped = task.file_set.justChunked()
    if test_mode:
        file_set_prepped = file_set_prepped.slice(chunks=slice(0,1))

    return file_set_prepped
