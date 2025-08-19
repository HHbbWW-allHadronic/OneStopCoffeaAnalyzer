from __future__ import annotations

import logging
import shutil
from pathlib import Path


import analyzer
from analyzer.configuration import CONFIG
from analyzer.utils.file_tools import compressDirectory

logger = logging.getLogger(__name__)

def setupForCondor(
    analysis_root_dir=None,
    apptainer_dir=None,
    venv_path=None,
    x509_path=None,
    temporary_path=None,
    extra_files=None,
):

    # print(f"{analysis_root_dir = }")
    # print(f"{apptainer_dir = }")
    # print(f"{venv_path = }")
    # print(f"{x509_path = }")
    # print(f"{temporary_path = }")
    extra_files = extra_files or []
    compressed_env = Path(CONFIG.APPLICATION_DATA) / "compressed" / "environment.tar.gz"
    analyzer_compressed = (
        Path(CONFIG.APPLICATION_DATA) / "compressed" / "analyzer.tar.gz"
    )

    if venv_path:
        if not compressed_env.exists():
            compressDirectory(
                input_dir=".application_data/venv",
                root_dir=analysis_root_dir,
                output=compressed_env,
                archive_type="gztar",
            )
    compressDirectory(
        input_dir=Path(analyzer.__file__).parent.relative_to(analysis_root_dir),
        root_dir=analysis_root_dir,
        output=analyzer_compressed,
        archive_type="gztar",
    )

    transfer_input_files = ["setup.sh", compressed_env, analyzer_compressed]

    if extra_files:
        extra_compressed = (
            Path(CONFIG.APPLICATION_DATA) / "compressed" / "extra_files.tar.gz"
        )
        transfer_input_files.append(extra_compressed)
        temp = Path(temporary_path)
        extra_files_path = temp / "extra_files/"
        extra_files_path.mkdir(exist_ok=True, parents=True)
        for i in extra_files:
            src = Path(i)
            shutil.copytree(src, extra_files_path / i)

        compressDirectory(
            input_dir="",
            root_dir=extra_files_path,
            output=extra_compressed,
            archive_type="gztar",
        )
    # if x509_path:
    #     transfer_input_files.append(x509_path)

    return transfer_input_files
