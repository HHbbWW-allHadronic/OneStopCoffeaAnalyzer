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

if CONFIG.PRETTY_MODE:
    from rich import print
    from rich.progress import track

class RegionAnalyzer(BaseModel):
    region_name: str
    description: str = ""
    forbid_data: bool = False

    preselection: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    corrections: list[ConfiguredAnalyzerModule] = Field(default_factory=list)

    objects: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    selection: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    categories: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    histograms: list[ConfiguredAnalyzerModule] = Field(default_factory=list)
    weights: list[ConfiguredAnalyzerModule] = Field(default_factory=list)

    @staticmethod
    def fromRegion(region_desc, sample, module_repo, era_repo):
        name = region_desc.name
        sample_params = sample.params
        dataset_params = sample_params.dataset
        dataset_params.populateEra(era_repo)
        if region_desc.forbid_data and dataset_params.sample_type == "Data":
            raise AnalysisConfigurationError(
                f"Region '{region_desc.name}' is marked with 'forbid_data'"
                f"but is recieving Data sample '{sample.name}'"
            )

        def resolveModules(l, t):
            ret = [
                mod
                for mod in l
                if not mod.sample_spec or mod.sample_spec.passes(dataset_params)
            ]
            modules = [
                [
                    module_repo.get(t, mod.name, configuration=c)
                    for c in (
                        mod.config if isinstance(mod.config, list) else [mod.config]
                    )
                ]
                for mod in ret
            ]
            ret = list(it.chain(*modules))
            return ret

        preselection = resolveModules(region_desc.preselection, ModuleType.Selection)
        corrections = resolveModules(region_desc.corrections, ModuleType.Producer)
        objects = resolveModules(region_desc.objects, ModuleType.Producer)

        selection = resolveModules(region_desc.selection, ModuleType.Selection)

        weights = resolveModules(region_desc.weights, ModuleType.Weight)
        categories = resolveModules(region_desc.categories, ModuleType.Categorization)

        preselection_histograms = resolveModules(
            region_desc.preselection_histograms, ModuleType.Histogram
        )
        postselection_histograms = resolveModules(
            region_desc.histograms, ModuleType.Histogram
        )

        return RegionAnalyzer(
            region_name=name,
            description=region_desc.description,
            forbid_data=region_desc.forbid_data,
            preselection=preselection,
            objects=objects,
            corrections=corrections,
            selection=selection,
            histograms=postselection_histograms,
            categories=categories,
            weights=weights,
        )

    def getSectorParams(self, sample_params):
        return SubSectorParams(sample=sample_params, region_name=self.region_name)

    def runPreselection(self, events, params, selection_set=None):
        params = self.getSectorParams(params)

        if selection_set is None:
            selection_set = SelectionSet()
        selection = Selection(select_from=selection_set)

        selector = Selector(selection, selection_set)
        for module in self.preselection:
            module(events, params, selector)
        return selection

    def runSelection(self, columns, params, selection_set=None):
        params = self.getSectorParams(params)
        if selection_set is None:
            selection_set = SelectionSet()
        selection = Selection(select_from=selection_set)
        selector = Selector(selection, selection_set)
        for module in self.selection:
            module(columns, params, selector)

        return selection

    def runCorrections(self, events, params, columns=None):
        params = self.getSectorParams(params)
        if columns is None:
            columns = Columns(events)
        for module in self.corrections:
            module(columns, params)
        return columns

    def runObjects(self, columns, params):
        params = self.getSectorParams(params)
        for module in self.objects:
            module(columns, params)
        return columns

    def runPostSelection(self, columns, params, histogram_storage):
        params = self.getSectorParams(params)
        active_shape = columns.syst
        weighter = Weighter(ignore_systematics=active_shape is not None)

        categories = []
        for module in self.weights:
            module(columns, params, weighter)
        for module in categories:
            module(columns, params)
        histogrammer = Histogrammer(
            storage=histogram_storage,
            weighter=weighter,
            categories=categories,
            active_shape_systematic=active_shape,
        )
        for module in self.histograms:
            module(columns, params, histogrammer)

    # def run(self, columns, params, variation=None):
    #     shape_columns = ColumnShapeSyst(columns, variation=variation)
    #     for module in self.corrections:
    #         module(events, params, shape_columns)
    #     return shape_columns


__subsector_param_cache = {}
__sample_param_cache = {}

def getParamsForSubSector(subsector_id, dataset_repo, era_repo):
    if subsector_id in __subsector_param_cache:
        return __subsector_param_cache[subsector_id]

    sample_id = subsector_id.sample_id
    dataset = dataset_repo[sample_id.dataset_name]
    params = dataset.getSample(sample_id.sample_name).params
    dataset_params = params.dataset
    dataset_params.populateEra(era_repo)
    sector_params = SectorParams(
        dataset=dataset_params,
        region={"region_name": subsector_id.region_name},
    )
    p = SubSectorParams(
        sector=sector_params,
        sample=params.sample,
        subsector_id=subsector_id,
    )
    __subsector_param_cache[subsector_id] = p
    return p


def getParamsSample(sample_id, dataset_repo, era_repo):
    if sample_id in __sample_param_cache:
        return __subsector_param_cache[subsector_id]

    dataset = dataset_repo[sample_id.dataset_name]
    params = dataset.getSample(sample_id.sample_name).params
    params.dataset.populateEra(era_repo)
    return params
