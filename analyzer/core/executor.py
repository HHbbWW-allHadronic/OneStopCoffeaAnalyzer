import copy
import enum
import inspect
import itertools as it
import logging
import pickle as pkl
import traceback
import operator as op
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union
import functools as ft

import yaml

import awkward as ak
import dask
from analyzer.configuration import CONFIG
from analyzer.datasets import DatasetRepo, EraRepo, SampleId, SampleType
from analyzer.utils.file_tools import extractCmsLocation
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import coffea.dataset_tools as cdt
from coffea.util import decompress_form
from pydantic import BaseModel, ConfigDict, Field

from .analysis_modules import (
    MODULE_REPO,
    AnalyzerModule,
    ModuleType,
    ConfiguredAnalyzerModule,
)
from .common_types import Scalar
from .histograms import HistogramSpec, HistogramCollection, Histogrammer
from .specifiers import SampleSpec, SubSectorId, SectorParams, SubSectorParams
from .columns import Column, Columns
from .selection import SelectionFlow, Selection, SelectionSet, Selector
import analyzer.core.results as results
from .weights import Weighter

class CondorExecutor(BaseModel):
    pass


class DaskExecutor:
    def run(self, analyzer, params, filesets):
        all_events = {}
        for sample_id, sample_chunks in sample_chunks:
            events, report = NanoEventsFactory.from_root(
                chunk_list,
                schemaclass=NanoAODSchema,
                uproot_options=dict(
                    allow_read_errors_with_report=True,
                    timeout=30,
                ),
                # known_base_form=maybe_base_form,
            ).events()
            all_events[sample_id] = (events, report)

        ret = analyzer.run(events, params)
        to_compute = {x: y.model_dump() for x, y in ret.items()}
        ret = dask.compute(to_compute)[0]
        ret = {x: results.SubSectorResult(**y) for x, y in ret.items()}
        return ret


class ImmediateExecutor:
    pass
