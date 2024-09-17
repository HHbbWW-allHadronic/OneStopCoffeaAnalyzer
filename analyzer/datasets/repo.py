import itertools as it
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import yaml

import rich
from analyzer.core.inputs import AnalyzerInput
from coffea.dataset_tools.preprocess import DatasetSpec
from rich.table import Table

from .samples import Dataset


