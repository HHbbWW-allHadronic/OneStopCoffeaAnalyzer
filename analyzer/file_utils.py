import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml

logger = logging.getLogger(__name__)


def compressDirectory(
    base_dir, zip_path=None, name="environment", archive_type="gztar"
):
    base_dir = Path(base_dir)
    base_name = base_dir.stem
    if not zip_path:
        temp_path = Path(tempfile.gettempdir())
    else:
        temp_path = Path(zip_path)

    trimmed_path = temp_path / f"temp_{base_name}" / base_name
    if trimmed_path.is_dir():
        logger.info(f"Deleting tree at {trimmed_path}")
        shutil.rmtree(trimmed_path)

    logger.info(f"Using {trimmed_path} as copy location.")
    temp_analyzer = shutil.copytree(
        base_dir,
        trimmed_path,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*~", "*.md"),
    )
    package_path = shutil.make_archive(
        temp_path / name,
        archive_type,
        root_dir=trimmed_path.parent,
        base_dir=base_name,
    )
    shutil.rmtree(trimmed_path.parent)

    final_path = temp_path / f"{name}.{archive_type}"
    logger.info(f"Created analyzer archive at {final_path}")
    return final_path


def copyFile(fr, to):
    fr_scheme, fr_netloc, fr_path, *fr_rest = urlparse(str(fr))
    to_scheme, to_netloc, to_path, *to_rest = urlparse(str(to))
    if not fr_scheme:
        fr_path = str(Path(fr_path).resolve().absolute())
    if not to_scheme:
        to_path = str(Path(to_path).resolve().absolute())
    fr = urlunparse((fr_scheme, fr_netloc, fr_path, *fr_rest))
    to = urlunparse((to_scheme, to_netloc, to_path, *to_rest))
    xrootd = any(x == "root" for x in (fr_scheme, to_scheme))
    if xrootd:
        import XRootD
        import XRootD.client

        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(str(fr), str(to))
        copyproc.prepare()
        copyproc.run()
        client = XRootD.client.FileSystem(to_netloc)
        status = client.locate(to_path, XRootD.client.flags.OpenFlags.READ)
        assert status[0].ok
        del client
        del copyproc
    else:
        to = Path(to)
        if not to.parent.is_dir():
            to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fr, to)


def appendToUrl(url, *args):
    scheme, netloc, path, *rest = urlparse(str(url))
    path = Path(path, *args)
    return urlunparse((scheme, netloc, str(path), *rest))


def getStem(url):
    scheme, netloc, path, *fr_rest = urlparse(str(url))
    return str(Path(path).stem)


class DirectoryData:
    data_file_name = "directory_data.yaml"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory_data_file = self.directory / self.data_file_name

    def getAll(self):
        if self.directory_data_file.is_file():
            with open(self.directory_data_file, "r") as f:
                data = yaml.safe_load(f)
            return data
        else:
            return {}

    def __updateData(self, newdata):
        f=tempfile.NamedTemporaryFile(delete=False, mode="w", dir=self.directory)
        f.write(yaml.dump(newdata))
        Path(f.name).rename(self.directory_data_file)

    def get(self, path):
        return self.getAll()[path]

    def set(self, path, data):
        current_data = self.get(path)
        current_data[path] = data
        self.__updateData(current_data)

    def sync(self):
        current_data = self.getAll()
        for p in self.directory.glob("**/*"):
            if p.is_file() and p != self.directory_data_file:
                key = str(p.relative_to(self.directory))
                if key not in current_data:
                    current_data[key] = {}
        self.__updateData(current_data)
