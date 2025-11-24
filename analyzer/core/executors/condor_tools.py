from __future__ import annotations

import logging
import shutil
import analyzer
from analyzer.configuration import CONFIG
from pathlib import Path
from attrs import define
import platform
from analyzer.utils.file_tools import zipDirectory, getVomsProxyPath
from jinja2 import Environment, PackageLoader, select_autoescape

logger = logging.getLogger(__name__)


@define
class CondorPackage:
    container: str
    transfer_file_list: list[str]
    setup_script: str



SCRIPT_TEMPLATE = """
# GENERATED AUTOMATICALLY 
{% for item in files_to_unzip %}
unzip {{ item }}
{% endfor %}
ls -alhtr
export X509_USER_PROXY=$(realpath $(find . -iname 'x509*'))
echo $X509_USER_PROXY
source {{ venv_activate_path }}
which python3
"""


def createCondorPackage(
    container,
    venv_path,
    analyzer_path=None,
    extra_files=None,
):
    """
    Returns a tuple of the following elements
    1.
    """

    condor_temp_loc = Path(CONFIG.general.base_data_path) / CONFIG.condor.temp_location
    condor_temp_loc.mkdir(exist_ok=True, parents=True)
    extra_files = extra_files or []
    compressed_env = condor_temp_loc / "environment.tar.gz"
    analyzer_path = analyzer_path or analyzer.__file__
    compressed_analyzer = condor_temp_loc / "analyzer.tar.gz"
    # voms_path = Path(getVomsProxyPath())
    voms_path = Path("randompw")

    if not compressed_env.exists():
        zipDirectory(venv_path, compressed_env)
    zipDirectory(analyzer_path, compressed_analyzer)

    script_path = condor_temp_loc / "setup.sh"
    transfer_input_files = [script_path, compressed_env, compressed_analyzer, voms_path]
    files_to_unzip = [compressed_env, compressed_analyzer]
    env = Environment()
    template = env.from_string(SCRIPT_TEMPLATE)

    venv_activate_path = Path(venv_path) / "bin" / "activate"

    script = template.render(
        files_to_unzip=[str(x.name) for x in files_to_unzip],
        venv_activate_path=venv_activate_path,
    )
    with open(script_path, "w") as f:
        f.write(script)

    return CondorPackage(container, transfer_input_files, "setup.sh")
