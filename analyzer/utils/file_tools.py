import collections.abc
import logging
import pickle
import shutil
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from analyzer.configuration import CONFIG

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
    #parts = Path(p).parts
    parts = p.split('/')
    root_idx = None
    for r in CONFIG.FILE_ROOTS:
        m = multiMatch(parts, r)
        if m is not None:
            root_idx = m
            break
    if root_idx is None:
        raise RuntimeError(f"Could not find 'store' in {parts}")
    good_parts = parts[root_idx:]
    #cms_path = Path(*good_parts)
    cms_path = '/'.join(good_parts)
    return str(cms_path)

# def extractCmsLocation(url):
#     _, _, p, *rest = urlparse(url)
#     parts = Path(p).parts
#     root_idx = None
#     for r in CONFIG.FILE_ROOTS:
#         m = multiMatch(parts, r)
#         if m is not None:
#             root_idx = m
#             break
#     if root_idx is None:
#         raise RuntimeError(f"Could not find 'store' in {parts}")
#     good_parts = parts[root_idx:]
#     cms_path = Path(*good_parts)
#     return str(cms_path)


def pickleWithParents(outpath, data):
    p = Path(outpath)
    p.parent.mkdir(exist_ok=True, parents=True)
    with open(p, "wb") as f:
        pickle.dump(data, f)


def compressDirectory(
    input_dir,
    root_dir,
    output,
    archive_type="gztar",
    temporary_path=".temporary",
):
    logger.info(f"Compressing directory '{input_dir}' relative to '{root_dir}'")
    logger.info(f"Output is '{output}'")
    stem = output.name
    temp = Path(temporary_path)
    temp.parent.mkdir(exist_ok=True, parents=True)
    output.parent.mkdir(exist_ok=True, parents=True)
    # base_name = base_dir.stem
    # if not zip_path:
    #     temp_path = Path(tempfile.gettempdir())
    # else:
    #     temp_path = Path(zip_path)

    # trimmed_path = temp_path / f"temp_{base_name}" / base_name
    # if trimmed_path.is_dir():
    #     logger.info(f"Deleting tree at {trimmed_path}")
    #     shutil.rmtree(trimmed_path)

    # logger.info(f"Using {trimmed_path} as copy location.")
    # temp_analyzer = shutil.copytree(
    #     base_dir,
    #     trimmed_path,
    #     ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*~", "*.md"),
    # )
    package_path = shutil.make_archive(
        str(temp / stem),
        archive_type,
        root_dir=str(root_dir),
        base_dir=str(input_dir),
        verbose=True,
        dry_run=False,
        group=None,
        owner=None,
        logger=logger,
    )
    logger.info(f"Created analyzer archive at {package_path}")
    shutil.copy(package_path, output)
    shutil.rmtree(temp)
    # final_path = temp_path / f"{name}.{archive_type}"
    return output


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
