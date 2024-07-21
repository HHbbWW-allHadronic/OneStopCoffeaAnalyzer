from functools import singledispatch
from typing import Any, Dict, Optional, Set, Tuple, Union
from .inputs import DatasetPreprocessed, AnalyzerInput

from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory


@singledispatch
def getEvents(arg, known_form=None, cache=None):
    events, report = NanoEventsFactory.from_root(
        arg,
        schemaclass=NanoAODSchema,
        uproot_options=dict(
            allow_read_errors_with_report=True,
            timeout=30,
        ),
        known_base_form=known_form,
        persistent_cache=cache,
    ).events()
    return events, report


@getEvents.register
def _(arg: AnalyzerInput, **kwargs):
    ds_pre = DatasetPreprocessed.fromDatasetInput(arg, **kwargs)
    return getEvents(ds_pre.coffea_dataset_split["files"])


@getEvents.register
def _(arg: str, sample_manager=None, **kwargs):
    from analyzer.datasets import SampleManager

    if "/" in arg:
        return getEvents({arg: "Events"})
    else:
        if isinstance(sample_manager, str):
            d = sample_manager
            sample_manager = SampleManager()
            sample_manager.loadSamplesFromDirectory(d)

        s = sample_manager.getSet(arg)
        return getEvents(s.getAnalyzerInput(), **kwargs)
