import argparse
import shutil
from pathlib import Path
import json
from pydantic import BaseModel
from analyzer.utils.file_tools import compressDirectory


def storeResults(directory, name):
    directory = Path(directory)
    output = Path(".temporary")
    output.mkdir(exist_ok=True, parents=True)
    archive_type="gztar"
    package_path = shutil.make_archive(
        (output / name)
        archive_type,
        root_dir=str(directory),
        base_dir=str(directory.parent),
        verbose=True,
        dry_run=False,
        group=None,
        owner=None)
    return package_path



    


    



    


 
    


