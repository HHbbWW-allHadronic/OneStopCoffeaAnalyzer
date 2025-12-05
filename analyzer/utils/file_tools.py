import collections.abc
import logging
import pickle
import os
from analyzer.utils.pretty import progbar
import shutil
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from analyzer.configuration import CONFIG

logger = logging.getLogger("analyzer")


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


def multiMatch(l, elements):
    if not isinstance(elements, (tuple, list)):
        elements = (elements,)
    else:
        elements = tuple(elements)
    n = len(elements)
    for i in range(len(l) - n + 1):
        if elements == tuple(l[i : i + n]):
            return i + n - 1
    return None


def extractCmsLocation(url):
    _, _, p, *rest = urlparse(url)
    # parts = Path(p).parts
    parts = p.split("/")
    root_idx = None
    for r in CONFIG.FILE_ROOTS:
        m = multiMatch(parts, r)
        if m is not None:
            root_idx = m
            break
    if root_idx is None:
        raise RuntimeError(f"Could not find 'store' in {parts}")
    good_parts = parts[root_idx:]
    # cms_path = Path(*good_parts)
    cms_path = "/".join(good_parts)
    return str(cms_path)


def zipDirectory(
    path,
    output,
    skip_words=(".git", ".github", ".pytest_cache", "tests", "docs"),
    skip=(lambda fn: os.path.splitext(fn)[1] == ".pyc",),
):
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in progbar(os.walk(path)):
            for file in files:
                filename = os.path.join(root, file)
                if any(predicate(filename) for predicate in skip):
                    continue
                dirs = filename.split(os.sep)
                if any(word in dirs for word in skip_words):
                    continue

                archive_name = os.path.relpath(
                    os.path.join(root, file), os.path.join(path, "..")
                )
                z.write(filename, archive_name)


def tarDirectory(
    path,
    output,
    skip_words=(".git", ".github", ".pytest_cache", "tests", "docs"),
    skip=(lambda fn: os.path.splitext(fn)[1] == ".pyc",),
    mode="w",
):
    with tarfile.open(output, f"{mode}:gz") as z:
        for root, dirs, files in progbar(os.walk(path)):
            for file in files:
                filename = os.path.join(root, file)
                if any(predicate(filename) for predicate in skip):
                    continue
                dirs = filename.split(os.sep)
                if any(word in dirs for word in skip_words):
                    continue

                archive_name = os.path.relpath(
                    os.path.join(root, file), os.path.join(path, "..")
                )
                z.add(filename, archive_name)



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
        to_is_dir = to_path[-1] == "/"
        logger.info(
            f'Dest path "{str(to_path)}" does not exist and it will {"" if to_is_dir  else "NOT"} be treated as a directory'
        )
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
    status = client.copy(str(fr), str(to), force=True)[0]
    logger.info(status)
    assert status.ok
    del client


def getVomsProxyPath(check_ok=True):
    import subprocess

    if check_ok:
        res = subprocess.run(
            ["voms-proxy-info", "-exists", "-valid", "2:0"], check=True
        )
        if res.returncode:
            raise Exception(
                "VOMS ERROR: please run `voms-proxy-init -voms cms -rfc --valid 168:0`"
            )
    proxy = subprocess.check_output(["voms-proxy-info", "-path"], text=True).strip()
    return proxy
