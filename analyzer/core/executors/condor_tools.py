from __future__ import annotations

import logging
import analyzer
from analyzer.configuration import CONFIG
from pathlib import Path
from attrs import define
from analyzer.utils.file_tools import getVomsProxyPath, tarDirectory, tarFiles
from jinja2 import Environment

logger = logging.getLogger("analyzer")


@define
class CondorPackage:
    container: str
    transfer_file_list: list[str]
    setup_script: str


SCRIPT_TEMPLATE = """
# GENERATED AUTOMATICALLY 
echo "STARTING SETUP"
{% for item in files_to_unzip %}
echo "UNTARRING {{item}}"
tar zxf {{ item }}
{% endfor %}
ls -alhtr
export X509_USER_PROXY=$(realpath {{ voms_path_in_container }})
echo $X509_USER_PROXY
echo "ACTIVATING VENV {{ venv_activate_path }}"
source {{ venv_activate_path }}
echo "HERERHERHERHEHRHERHER"
which python3
"""


def createCondorPackage(
    container,
    venv_path,
    extra_files=None,
):
    condor_temp_loc = Path(CONFIG.general.base_data_path) / CONFIG.condor.temp_location
    condor_temp_loc.mkdir(exist_ok=True, parents=True)
    extra_files = extra_files or []
    compressed_env = condor_temp_loc / "environment.tar.gz"
    compressed_extra = condor_temp_loc / "extras.tar.gz"
    compressed_analyzer = condor_temp_loc / "analyzer.tar.gz"

    analyzer_path = Path(analyzer.__file__).parent

    voms_path = Path(getVomsProxyPath())
    voms_path_in_container = voms_path.name

    if not compressed_env.exists():
        logger.info(f"Did not find {compressed_env}, creating compressed directory.")
        logger.info(
            "Creating compressed virtual environment. This needs to be done only once."
        )
        tarDirectory(venv_path, compressed_env)

    logger.info("Creating compressed analyzer")
    compressed_extra.unlink(missing_ok=True)
    if extra_files is not None:
        tarFiles(extra_files, compressed_extra)

    tarDirectory(analyzer_path, compressed_analyzer)

    script_path = condor_temp_loc / "setup.sh"
    transfer_input_files = [
        script_path,
        compressed_env,
        compressed_analyzer,
        str(voms_path),
    ]

    files_to_unzip = [compressed_env, compressed_analyzer]

    if extra_files is not None:
        transfer_input_files.append(compressed_extra)
        files_to_unzip.append(compressed_extra)

    env = Environment()
    template = env.from_string(SCRIPT_TEMPLATE)

    venv_activate_path = Path(venv_path) / "bin" / "activate"

    script = template.render(
        files_to_unzip=[str(x.name) for x in files_to_unzip],
        venv_activate_path=venv_activate_path,
        voms_path_in_container=voms_path_in_container,
    )
    with open(script_path, "w") as f:
        f.write(script)

    return CondorPackage(container, transfer_input_files, "setup.sh")
