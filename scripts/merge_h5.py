#!/usr/bin/env python3
"""
Usage:
    python merge_h5.py --config merge_config.yaml [--dry-run]
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import fsspec
import h5py
import numpy as np
import yaml

logger = logging.getLogger(__name__)



def xrd_listdir(xrd_dir: str) -> list[str]:
    """Return basenames of all files in an XRootD directory."""
    parsed = urlparse(xrd_dir)
    fs = fsspec.filesystem("root", hostid=parsed.netloc)
    entries = fs.ls(parsed.path, detail=False)
    return [Path(e).name for e in entries]


def xrd_open_h5(xrd_url: str) -> h5py.File:
    """Open a remote HDF5 file directly via fsspec — no local staging."""
    f = fsspec.open(xrd_url, "rb")
    return h5py.File(f.open(), "r")



def discover_structure(h5file: h5py.File) -> dict[str, dict]:
    """Walk one file to get dataset paths, dtypes, and trailing shapes."""
    meta = {}
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim > 0:
            meta[name] = {"dtype": obj.dtype, "shape": obj.shape[1:]}
    h5file.visititems(_visit)
    return meta


def merge_group(
    group_cfg: dict,
    source_dir: str,
    all_files: list[str],
    chunk_size: int,
    output_dir: Path,
    dry_run: bool,
) -> None:
    name       = group_cfg["name"]
    pattern    = re.compile(group_cfg["pattern"])
    chunk_size = group_cfg.get("chunk_size", chunk_size)

    matched = sorted(f for f in all_files if pattern.search(f))

    if not matched:
        logger.warning("[%s] No files matched pattern %r — skipping.", name, group_cfg["pattern"])
        return

    logger.info("[%s] Matched %d file(s):", name, len(matched))
    for f in matched:
        logger.info("    %s", f)

    if dry_run:
        logger.info("[%s] Dry-run mode — no output written.", name)
        return

    output_path = output_dir / f"{name}.h5"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_base = urlparse(source_dir)

    def file_url(fname: str) -> str:
        file_path = str(Path(src_base.path.rstrip("/")) / fname)
        return urlunparse(src_base._replace(path=file_path))

    # Copy structure of first file for all
    with xrd_open_h5(file_url(matched[0])) as hf:
        ds_meta = discover_structure(hf)
        row_counts = {ds: hf[ds].shape[0] for ds in ds_meta}

    for fname in matched[1:]:
        with xrd_open_h5(file_url(fname)) as hf:
            for ds in ds_meta:
                row_counts[ds] += hf[ds].shape[0]

    n_rows = row_counts[next(iter(ds_meta))]

    logger.info("[%s] Total rows: %d across %d dataset(s):", name, n_rows, len(ds_meta))
    for ds_path, meta in ds_meta.items():
        logger.info("    %-30s  dtype=%s  trailing_shape=%s", ds_path, meta["dtype"], meta["shape"])

    with h5py.File(output_path, "w") as out:

        handles: dict[str, h5py.Dataset] = {}
        for ds_path, meta in ds_meta.items():
            handles[ds_path] = out.create_dataset(
                ds_path,
                shape=(n_rows,) + meta["shape"],
                dtype=meta["dtype"],
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )

        offset = 0
        for fname in matched:
            with xrd_open_h5(file_url(fname)) as hf:
                n = hf[next(iter(ds_meta))].shape[0]

                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    for ds_path in ds_meta:
                        handles[ds_path][offset + start : offset + end] = hf[ds_path][start:end]

                offset += n

        out.attrs["source_file_count"] = len(matched)
        out.attrs["source_pattern"]    = group_cfg["pattern"]
        out.attrs["source_dir"]        = source_dir

    logger.info("[%s] Written → %s", name, output_path)



def main() -> None:
    parser = argparse.ArgumentParser(description="Merge XRootD HDF5 files per YAML config.")
    parser.add_argument("--config",    default="merge_config.yaml")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--groups",    nargs="*", metavar="GROUP")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, args.log_level))

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    source_dir = cfg["source_dir"]
    output_dir = Path(cfg["output_dir"])
    chunk_size = cfg.get("chunk_size", 16000)

    logger.info("Listing files in %s …", source_dir)
    all_files = xrd_listdir(source_dir)
    logger.info("Found %d file(s) total.", len(all_files))

    groups = cfg.get("groups", [])

    for group_cfg in groups:
        merge_group(
            group_cfg  = group_cfg,
            source_dir = source_dir,
            all_files  = all_files,
            chunk_size = chunk_size,
            output_dir = output_dir,
            dry_run    = args.dry_run,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
