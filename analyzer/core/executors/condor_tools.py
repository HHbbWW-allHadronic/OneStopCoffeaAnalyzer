from __future__ import annotations

import logging
import shutil
import analyzer
from pathlib import Path
from attrs import define
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

    extra_files = extra_files or []
    compressed_env = Path(".application_data") / "condor" / "environment.tar.gz"
    analyzer_path = analyzer_path or analyzer.__file__
    analyzer_compressed = Path(".application_data") / "condor" / "analyzer.tar.gz"
    voms_path = Path(getVomsProxyPath())

    if not compressed_env.exists():
        zipDirectory(venv_path, compressed_env)
    zipDirectory(analyzer_path, compressed_analyzer)

    script_path = Path(".application_data" / "condor" / "setup.sh")
    transfer_input_files = [script_path, compressed_env, analyzer_compressed, voms_path]
    env = Environment()
    template = env.from_string(SCRIPT_TEMPLATE)

    venv_activate_path = Path(venv_path).name / "bin" / "activate"

    script = template.render(
        files_to_unzip=[str(x.name) for x in transfer_input_files],
        venv_activate_path=venv_activate_path,
    )
    with open(script_path, "w") as f:
        f.write(script)

    return CondorPackage(container, transfer_input_files, "setup.sh")
