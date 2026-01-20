from __future__ import annotations

import itertools as it
import functools as ft
from pathlib import Path
from attrs import define
import uproot
import os
import operator as op
import jinja2
from analyzer.utils.querying import BasePattern

from analyzer.utils.structure_tools import (
    commonDict,
    dictToDot,
    doFormatting,
)
from .processors import BasePostprocessor


def formatLines(elems, separator=" ", force_max=None):
    elems = [[str(x) for x in y] for y in elems]
    max_lens = [force_max or max(len(x) for x in y) for y in zip(*elems)]
    max_lens = [force_max or max(len(x) for x in y) for y in zip(*elems)]
    row_format = separator.join(f"{{: <{length}}}" for length in max_lens)
    return [row_format.format(*e) for e in elems]


@define(frozen=True)
class Process:
    name: str
    is_signal: bool = False


@define(frozen=True)
class Systematic:
    name: str
    dist: str


@define(frozen=True)
class Channel:
    name: str


class DataCard:
    def __init__(self):
        self.processes = []
        self.systematics = []
        self.channels = []

        self.observations = {}
        self.process_systematics = {}
        self.process_rates = {}
        self.process_shapes = {}
        self.process_shape_systematics = {}

    def addProcess(self, process: Process):
        self.processes.append(process)

    def addSystematic(self, systematic: Systematic):
        self.systematics.append(systematic)

    def addChannel(self, channel: Channel):
        self.channels.append(channel)

    def setProcessSystematic(
        self, process: Process, systematic: Systematic, channel: Channel, value: float
    ):
        self.process_systematics[(channel, process, systematic)] = value

    def setProcessRate(self, process: Process, channel: Channel, value: float):
        self.process_rates[(channel, process)] = value

    def addShape(
        self,
        process: Process,
        channel: Channel,
        root_file,
        shape_name,
        shape_systematic_name,
    ):
        self.process_shapes[(channel, process)] = (
            root_file,
            shape_name,
            shape_systematic_name,
        )

    def addObservation(
        self,
        channel: Channel,
        root_file,
        hist_name,
        value,
    ):
        self.observations[channel] = (root_file, hist_name, value)

    def constructHeader(self):
        lines = []
        lines.append("# Autodatacard")
        lines.append(f"imax {len(self.channels)}")
        lines.append(f"jmax {len(self.processes) - 1}")
        lines.append(f"kmax {len(self.systematics)}")
        return lines

    def constructShapes(self):
        rows = []
        for (channel, process), (
            root_file,
            shape_name,
            shape_syst_name,
        ) in self.process_shapes.items():
            row = [
                "shapes",
                process.name,
                channel.name,
                root_file,
                shape_name,
                shape_syst_name,
            ]
            row = [x for x in row if x is not None]
            rows.append(row)

        for channel, (root_file, shape_name, value) in self.observations.items():
            row = ["shapes", "data_obs", channel.name, root_file, shape_name, ""]
            row = [x for x in row if x is not None]
            rows.append(row)

        return formatLines(rows, separator="  ")

    def constructObservations(self):
        cols = [["bin", "observation"]]
        for channel, (root_file, shape_name, value) in self.observations.items():
            cols.append([channel.name, "-1"])
        rows = list(zip(*cols))
        lines = formatLines(rows, separator="  ")
        return lines

    def constructSystematics(self):
        processes = enumerate(
            reversed(sorted(self.processes, key=lambda x: x.is_signal))
        )
        cols = []
        first_col = [
            "bin",
            "process",
            "process",
            "rate",
            *(x.name for x in self.systematics),
        ]
        second_col = ["", "", "", "", *(x.dist for x in self.systematics)]
        cols.append(first_col)
        cols.append(second_col)
        for channel, (i, process) in sorted(
            it.product(self.channels, processes), key=lambda x: x[0].name
        ):
            current_col = [
                channel.name,
                process.name,
                i,
                self.process_rates[(channel, process)],
            ]
            for systematic in self.systematics:
                s = self.process_systematics.get((channel, process, systematic), None)
                syst_string = str(s) if s is not None else "-"
                current_col.append(syst_string)
            cols.append(current_col)
        rows = list(zip(*cols))
        lines = formatLines(rows, separator="  ")
        lines.insert(4, "#" * len(lines[0]))
        return lines

    def dumps(self):
        lines = self.constructHeader()
        lines += [""] * 2
        lines += self.constructShapes()
        lines += [""] * 2
        lines += self.constructObservations()
        lines += [""] * 2
        lines += self.constructSystematics()
        output = "\n".join(lines) + "\n"
        return output


def _handleProcessSystematics(
    rf,
    card,
    hist,
    process_name,
    channel,
    proc,
    systematics_pattern,
    root_file_name,
):
    if hasattr(hist.axes, "name") and "variation" in hist.axes.name:
        import re

        syst_ax = hist.axes["variation"]
        for syst_idx, syst_name in enumerate(syst_ax):
            if syst_name == "nominal":
                continue

            if systematics_pattern and not re.search(systematics_pattern, syst_name):
                continue

            if "Up" in syst_name or "Down" in syst_name:
                base_syst_name = syst_name.replace("Up", "").replace("Down", "")
                syst_obj = Systematic(base_syst_name, "shape")
                if syst_obj not in card.systematics:
                    card.addSystematic(syst_obj)

                h_var = hist[{"systematic": syst_name}]
                rf[f"{process_name}_{syst_name}"] = h_var
                card.setProcessSystematic(proc, syst_obj, channel, 1.0)


def _processList(
    rf,
    card,
    items,
    channel,
    root_file_name,
    is_signal,
    default_process_name,
    systematics_pattern=None,
):
    for item in items:
        meta = item.metadata
        hist = item.item.histogram
        process_name = meta.get("dataset_name", default_process_name)
        proc = Process(process_name, is_signal=is_signal)
        if proc not in card.processes:
            card.addProcess(proc)

        # Get nominal
        if hasattr(hist.axes, "name") and "variation" in hist.axes.name:
            if "nominal" in hist.axes["variation"]:
                h_nom = hist[{"variation": "nominal"}]
            else:
                h_nom = hist[{"variation": 0}]
            rate = h_nom.sum().value
            rf[process_name] = h_nom
        else:
            rate = hist.sum().value
            rf[process_name] = hist

        card.setProcessRate(proc, channel, rate)
        card.addShape(
            proc, channel, root_file_name, process_name, f"{process_name}_$SYSTEMATIC"
        )

        _handleProcessSystematics(
            rf,
            card,
            hist,
            process_name,
            channel,
            proc,
            systematics_pattern,
            root_file_name,
        )


def _handleObservation(rf, card, observation, background, channel, root_file_name):
    if observation:
        if len(observation) != 1:
            raise ValueError("Observation must have exactly one element.")

        rf["data_obs"] = observation[0].item.histogram
        card.addObservation(channel, root_file_name, "data_obs", -1)
    else:
        background_hists = [item.item.histogram for item in background]
        if background_hists:
            hist = ft.reduce(op.add, background_hists)

            if hasattr(hist.axes, "name") and "variation" in hist.axes.name:
                if "nominal" in hist.axes["variation"]:
                    hist = hist[{"variation": "nominal"}]
                else:
                    hist = hist[{"variation": 0}]
            rf["data_obs"] = hist
            card.addObservation(channel, root_file_name, "data_obs", -1)


def writeDatacard(
    channel_name,
    signal,
    background,
    common_metadata,
    output_dir,
    observation=None,
    systematics_pattern=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    card = DataCard()

    channel = Channel(channel_name)
    card.addChannel(channel)

    root_file_name = f"shapes_{channel_name}.root"
    root_file_path = output_dir / root_file_name

    with uproot.recreate(root_file_path) as rf:
        _processList(
            rf,
            card,
            signal,
            channel,
            root_file_name,
            is_signal=True,
            default_process_name="signal",
            systematics_pattern=systematics_pattern,
        )
        _processList(
            rf,
            card,
            background,
            channel,
            root_file_name,
            is_signal=False,
            default_process_name="background",
            systematics_pattern=systematics_pattern,
        )

    with uproot.update(root_file_path) as rf:
        _handleObservation(rf, card, observation, background, channel, root_file_name)

    datacard_content = card.dumps()
    datacard_path = output_dir / f"datacard_{channel_name}.txt"
    with open(datacard_path, "w") as f:
        f.write(datacard_content)

    return datacard_path


def writeCountingExperimentScript(output_dir, datacard_paths):
    template_str = """#!/bin/bash

# Auto-detected container execution
RUN_CMD=""

if command -v combine &> /dev/null; then
    echo "Using system 'combine'..."
    RUN_CMD=""
elif command -v apptainer &> /dev/null && [ -d "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest" ]; then
    echo "Using Apptainer with CVMFS container..."
    export APPTAINER_CACHEDIR="/tmp/$(whoami)/apptainer_cache"
    

    function run_combine() {
       IMAGE="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest"
       apptainer exec -B /cvmfs  $IMAGE /bin/bash -c 'source /cvmfs/cms.cern.ch/cmsset_default.sh && pushd $PWD && cd /home/cmsusr/CMSSW_14_1_0_pre4/ && eval $(scramv1 runtime -sh) && popd && exec \"$@\"' -- $@
    }
    RUN_CMD="run_combine"


elif command -v docker &> /dev/null; then
    echo "Using Docker container..."
    TAG="${COMBINE_TAG:-latest}"
    IMAGE="gitlab-registry.cern.ch/cms-cloud/combine-standalone:$TAG"
    RUN_CMD="docker run --rm -v $PWD:$PWD -w $PWD $IMAGE"
else
    echo "WARNING: No 'combine' found. Output commands may fail."
    RUN_CMD=""
fi

# Parse arguments
DO_MERGE=false
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -m|--merge)
      DO_MERGE=true
      shift
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
eval set -- "$PARAMS"

{% if datacards|length > 1 %}
if [ "$DO_MERGE" = true ]; then
    echo "Merging cards..."
    MERGED_CARD="merged_card.txt"
    $RUN_CMD combineCards.py {% for path in datacards %}{{ path.name }} {% endfor %} > $MERGED_CARD
    echo "Merged card created: $MERGED_CARD"

    echo "Running combine for Merged Card"
    $RUN_CMD combine -M Significance $MERGED_CARD > merged_significance.log
    $RUN_CMD combine -M AsymptoticLimits $MERGED_CARD > merged_limits.log
fi
{% endif %}

{% for path in datacards %}
echo 'Running combine for {{ path.name }}'
$RUN_CMD combine -M Significance {{ path.name }} | tee {{ path.stem }}_significance.log
$RUN_CMD combine -M AsymptoticLimits {{ path.name }} | tee {{ path.stem }}_limits.log
{% endfor %}
"""
    template = jinja2.Template(template_str)

    # Ensure they are Path objects for .name and .stem access
    paths = [Path(p) for p in datacard_paths]

    script_content = template.render(datacards=paths)

    script_path = Path(output_dir) / "run_combine.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)


@define
class CombineDatacard(BasePostprocessor):
    output_name: str
    channel: str
    systematics: str | None = None

    def getRunFuncs(self, group, prefix=None):
        signal = group["signal"]
        background = group["background"]
        observation = group.get("observation") or group.get("data")

        for sig in signal:
            current_signal = [sig]
            common_meta = commonDict(it.chain(current_signal, observation or []))
            formatted_output_dir = doFormatting(
                self.output_name, **dict(dictToDot(common_meta)), prefix=prefix
            )

            if self.channel:
                channel_name = doFormatting(
                    self.channel, **dict(dictToDot(common_meta))
                )
            else:
                channel_name = "bin1"

            yield ft.partial(
                createDatacardsAndScript,
                signal=current_signal,
                background=background,
                observation=observation,
                common_meta=common_meta,
                output_dir=formatted_output_dir,
                channel_name=channel_name,
                systematics_pattern=self.systematics,
            )


def createDatacardsAndScript(
    signal,
    background,
    common_meta,
    output_dir,
    observation=None,
    systematics_pattern=None,
    channel_name="bin1",
):
    datacard_path = writeDatacard(
        channel_name=channel_name,
        signal=signal,
        background=background,
        observation=observation,
        common_metadata=common_meta,
        output_dir=output_dir,
        systematics_pattern=systematics_pattern,
    )

    writeCountingExperimentScript(output_dir, [datacard_path])
