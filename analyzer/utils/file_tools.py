import collections.abc
import logging
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import pickle

import yaml

logger = logging.getLogger(__name__)


def stripPort(url):
    protocol, netloc, *rest = urlparse(url)
    netloc = netloc.split(":")[0]
    return urlunparse((protocol, netloc, *rest))


def stripPrefix(url):
    protocol, netloc, *rest = urlparse(url)
    netloc = netloc.split(":")[0]
    return urlunparse(("", "", *rest))


def getPath(url):
    _, _, p, *rest = urlparse(url)
    return Path(p)


def extractCmsLocation(url):
    _, _, p, *rest = urlparse(url)
    parts = Path(p).parts
    store_idx = next((i for i, x in enumerate(parts) if x == "store"), None)
    if store_idx is None:
        raise RuntimeError(f"Could not find 'store' in {parts}")

    good_parts = parts[store_idx:]
    cms_path = Path("/", *good_parts)
    return str(cms_path)


def pickleWithParents(outpath, data):
    p = Path(outpath)
    p.parent.mkdir(exist_ok=True, parents=True)
    with open(p, "wb") as f:
        pickle.dump(data, f)


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


def exists(client, loc):
    from XRootD.client.flags import OpenFlags

    status, result = client.stat(loc, OpenFlags.REFRESH)
    return status.ok


def isDir(client, loc):
    from XRootD.client.flags import OpenFlags, StatInfoFlags

    status, result = client.stat(loc, OpenFlags.REFRESH)
    return bool(status.ok and (result.flags & StatInfoFlags.IS_DIR))


def makeDir(client, loc):
    from XRootD.client.flags import MkDirFlags

    status, result = client.mkdir(loc, MkDirFlags.MAKEPATH)


def copyFile(fr, to, from_rel_to=None):
    logger.info(f'Dest path is "{str(to)}"')

    fr_scheme, fr_netloc, fr_path, *fr_rest = urlparse(str(fr))
    to_scheme, to_netloc, to_path, *to_rest = urlparse(str(to))
    if from_rel_to:
        rest = Path(fr_path).relative_to(Path(from_rel_to))
        to_path = str(Path(to_path) / rest)

    fr = urlunparse((fr_scheme, fr_netloc, str(Path(fr_path).absolute()), *fr_rest))
    to = urlunparse((to_scheme, to_netloc, to_path, *to_rest))

    xrootd = any("root" in x for x in (fr_scheme, to_scheme))

    import XRootD
    import XRootD.client

    client = XRootD.client.FileSystem(to_netloc)

    logger.info(f'Dest path is "{to_path}"')

    is_dir = isDir(client, to_path)
    ex = exists(client, to_path)

    if ex and not is_dir:
        raise RuntimeError(f"Destination exists and is not a directory")

    elif ex and is_dir:
        logger.info(f'Dest path "{to_path}" exists and is a directory')
        to = urlunparse(
            (to_scheme, to_netloc, str(Path(to_path) / Path(fr_path).name), *to_rest)
        )

    elif not ex:
        to_is_dir = to_path[-1] == '/'
        logger.info(f'Dest path "{str(to_path)}" does not exist and it will {"" if to_is_dir  else "NOT"} be treated as a directory')
        if to_is_dir:
            makeDir(client, str(to_path))
            logger.info(f'Creating directory  "{str(to_path)}"')
            dest = str(Path(to_path) / Path(fr_path).name)
            to = urlunparse((to_scheme, to_netloc, dest, *to_rest))
        else:
            parent_path = str(Path(to_path).parent)
            logger.info(f'Creating directory "{str(parent_path)}"')
            makeDir(client, str(parent_path))

    logger.info(f'FINAL DEST IS: "{to}"')
    # copyproc = XRootD.client.CopyProcess()
    # copyproc.add_job(str(fr), str(to))

    status = client.copy(str(fr), str(to), force=True)[0]
    logger.info(status)
    assert status.ok
    del client


def appendToUrl(url, *args):
    scheme, netloc, path, *rest = urlparse(str(url))
    path = Path(path, *args)
    return urlunparse((scheme, netloc, str(path), *rest))


def getStem(url):
    scheme, netloc, path, *fr_rest = urlparse(str(url))
    return str(Path(path).stem)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class DirectoryData:
    data_file_name = "directory_data.yaml"
    file_data_key = "file_data"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory_data_file = self.directory / self.data_file_name

    def getComplete(self):
        if self.directory_data_file.is_file():
            with open(self.directory_data_file, "r") as f:
                data = yaml.safe_load(f)
            return data
        else:
            return {}

    def __updateData(self, newdata):
        current_data = self.getComplete()
        new_dict = update(current_data, newdata)
        f = tempfile.NamedTemporaryFile(delete=False, mode="w", dir=self.directory)
        f.write(yaml.dump(new_dict))
        Path(f.name).rename(self.directory_data_file)

    def __key(self, path):
        k = str(Path(path).relative_to(self.directory))
        return k

    def get(self, path):
        k = self.__key(path)
        return self.getComplete()[self.file_data_key][k]

    def set(self, path, data):
        k = self.__key(path)
        d = {self.file_data_key: {k: data}}
        self.__updateData(d)

    def getGlobal(self):
        return self.getComplete()["global_data"]

    def setGlobal(self, data):
        self.__updateData({"global_data": data})

    def sync(self):
        current_data = self.getAll()
        for p in self.directory.glob("**/*"):
            if p.is_file() and p != self.directory_data_file:
                key = self.__key(path)
                if key not in current_data:
                    current_data[self.file_data_key][key] = {}
        self.__updateData(current_data)
