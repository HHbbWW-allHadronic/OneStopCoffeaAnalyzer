#!/usr/bin/env python3
"""
Merge the per-chunk ROOT skims written by SaveEvents into larger files with
haddnano, and compute the effective cross sections needed to normalize them.

SaveEvents writes one file per processed chunk, named
    {dataset}__{sample}__{file_id}__{event_start}_{event_stop}.root
even when no events pass the selection. For each (dataset, sample):

    N_processed = sum(event_stop - event_start) over ALL files (empty included)
    N_pass      = sum(Events entries)
    x_sec_eff   = x_sec * N_pass / N_processed

Empty files are not merged, but they still count toward N_processed.

One dataset yaml is written per original dataset yaml, with n_events = N_pass,
x_sec = x_sec_eff and the source replaced by the merged files, so that
mergeAndScale (lumi * x_sec / processed_events) gives lumi * x_sec / N_processed,
identical to running on the original NanoAOD. The summary CSV, the manifest
(inputs of every output) and the yamls are also copied to {output_dir}/_meta.

Outputs that already exist in output_dir are skipped, so a failed run can be
rerun to retry only what is missing.

Usage:
    python merge_root.py --config merge_root_config.yaml [--dry-run] [--samples REGEX ...]

--samples restricts merging to matching "{dataset}__{sample}" keys; bookkeeping
(summary, manifest, yamls) is only written by runs over all samples.
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import fsspec
import numpy as np
import uproot
import yaml

logger = logging.getLogger(__name__)

FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)__(?P<sample>.+?)__(?P<file_id>[0-9A-F]{32})__(?P<start>\d+)_(?P<stop>\d+)\.root$"
)

# Environment variables passed through to the ROOT subprocess
PASSTHROUGH_ENV = ["HOME", "USER", "LOGNAME", "KRB5CCNAME", "X509_USER_PROXY", "X509_CERT_DIR", "TMPDIR"]


def is_remote(path: str) -> bool:
    return path.startswith("root://")


def join_url(base: str, name: str) -> str:
    parsed = urlparse(base)
    return urlunparse(parsed._replace(path=parsed.path.rstrip("/") + "/" + name))


def xrd_listdir(xrd_dir: str) -> dict[str, int]:
    """Return {basename: size} for all files in an XRootD (or local) directory."""
    if not is_remote(xrd_dir):
        p = Path(xrd_dir)
        return {f.name: f.stat().st_size for f in p.iterdir() if f.is_file()} if p.exists() else {}
    parsed = urlparse(xrd_dir)
    fs = fsspec.filesystem("root", hostid=parsed.netloc)
    try:
        entries = fs.ls(parsed.path, detail=True)
    except FileNotFoundError:
        return {}
    return {Path(e["name"]).name: e["size"] for e in entries if e.get("type") == "file"}


def copy_out(local: Path, target: str) -> None:
    """Copy a local file to a local or XRootD destination, creating parent dirs.

    --posc makes EOS discard the file unless the copy completes, so a failed
    copy never leaves a partial output that a rerun would skip.
    """
    if not is_remote(target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        return
    parsed = urlparse(target)
    subprocess.run(["xrdfs", parsed.netloc, "mkdir", "-p", str(Path(parsed.path).parent)], check=True)
    subprocess.run(["xrdcp", "-f", "-s", "--posc", str(local), target], check=True)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_file(url: str) -> dict:
    """Entries, sum(sign(genWeight)) and sum(genWeight) of an Events tree."""
    with uproot.open(url, timeout=120) as f:
        if "Events" not in f:
            return {"entries": 0, "sign_sum": 0, "sumw": 0.0, "no_tree": True}
        t = f["Events"]
        n = int(t.num_entries)
        sign_sum, sumw = 0, 0.0
        if n and "genWeight" in t:
            gw = t["genWeight"].array(library="np")
            # Same convention as the genWeight sign correction in the analyzer
            sign_sum = int(np.where(gw > 0, 1, -1).sum())
            sumw = float(gw.sum())
        return {"entries": n, "sign_sum": sign_sum, "sumw": sumw}


def inspect_all(source_dir, names_sizes, cache_path, workers):
    """Inspect files in parallel, caching results keyed by filename and size.

    Uses processes, not threads: deserializing the ~3500-branch TTree metadata
    is CPU-bound Python and does not scale under the GIL.
    """
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    todo = [n for n, s in names_sizes.items() if cache.get(n, {}).get("size") != s]
    logger.info("Inspecting %d file(s) (%d cached) …", len(todo), len(names_sizes) - len(todo))

    def save():
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(cache_path)

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect_file, join_url(source_dir, n)): n for n in todo}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                info = fut.result()
            except Exception as e:  # corrupt / unreadable file
                logger.error("Could not read %s: %s", n, e)
                info = {"error": str(e)}
            info["size"] = names_sizes[n]
            cache[n] = info
            done += 1
            if done % 500 == 0:
                logger.info("  … %d / %d", done, len(todo))
                save()
    save()
    return {n: cache[n] for n in names_sizes}


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def load_original_yamls(patterns, base_dir):
    """Return ({path: datasets}, {(dataset_name, sample_name): sample_dict})."""
    yamls, samples = {}, {}
    for pat in patterns:
        pat = str(base_dir / pat) if not os.path.isabs(pat) else pat
        for path in sorted(glob.glob(pat)):
            with open(path) as fh:
                data = yaml.safe_load(fh) or []
            yamls[path] = data
            for ds in data:
                # Single-sample shorthand: sample fields live on the dataset
                for s in ds.get("samples", [ds]):
                    samples[(ds["name"], s["name"])] = s
    return yamls, samples


# Keys describing where a sample's events come from; replaced by `files`
SOURCE_KEYS = {"das_path", "files", "tree_name", "schema_name", "source"}


def update_sample(s, new):
    """Swap the source for the merged files and set n_events / x_sec in place."""
    for k in SOURCE_KEYS:
        s.pop(k, None)
    s["n_events"] = new["n_events"]
    if new.get("x_sec") is not None:
        s["x_sec"] = new["x_sec"]
    s["files"] = new["files"]


def write_skim_yamls(yamls, new_values, out_dir, suffix):
    """One yaml per original yaml, keeping only skimmed samples with new values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path, datasets in yamls.items():
        out = []
        for ds in datasets:
            ds = deepcopy(ds)
            if "samples" in ds:
                ds["samples"] = [s for s in ds["samples"] if (ds["name"], s["name"]) in new_values]
                if not ds["samples"]:
                    continue
                for s in ds["samples"]:
                    update_sample(s, new_values[(ds["name"], s["name"])])
            else:
                if (ds["name"], ds["name"]) not in new_values:
                    continue
                update_sample(ds, new_values[(ds["name"], ds["name"])])
            ds["name"] = ds["name"] + suffix
            out.append(ds)
        if out:
            target = out_dir / Path(path).name
            with open(target, "w") as fh:
                yaml.safe_dump(out, fh, sort_keys=False, width=1000)
            written.append(target)
    return written


def check_overlaps(files):
    """Warn about overlapping chunk ranges from the same input file."""
    by_id = defaultdict(list)
    for f in files:
        by_id[f["file_id"]].append((f["start"], f["stop"], f["name"]))
    n_bad = 0
    for ranges in by_id.values():
        ranges.sort()
        for (s0, e0, n0), (s1, e1, n1) in zip(ranges, ranges[1:]):
            if s1 < e0:
                n_bad += 1
                logger.warning("Overlapping chunks: %s and %s", n0, n1)
    return n_bad


def make_chunks(files, target_bytes, max_events, max_files):
    """Greedily pack non-empty files into chunks bounded by size/events/count."""
    chunks, cur, cur_b, cur_n = [], [], 0, 0
    for f in files:
        over = (
            (target_bytes and cur_b + f["size"] > target_bytes)
            or (max_events and cur_n + f["entries"] > max_events)
            or (max_files and len(cur) >= max_files)
        )
        if cur and over:
            chunks.append(cur)
            cur, cur_b, cur_n = [], 0, 0
        cur.append(f)
        cur_b += f["size"]
        cur_n += f["entries"]
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def clean_env():
    env = {k: os.environ[k] for k in PASSTHROUGH_ENV if k in os.environ}
    env["PATH"] = "/usr/bin:/bin"
    return env


def run_haddnano(out: Path, inputs, cfg, log) -> None:
    cmd = " ".join([f"source {cfg['root_setup']} &&", "python", cfg["haddnano"], str(out), *inputs])
    res = subprocess.run(["bash", "-c", cmd], env=clean_env(), stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f"haddnano failed for {out.name} (see {log.name})")


def merge_chunk(inputs, out_name, expected, cfg, staging_dir, log_dir):
    """haddnano the inputs into staging, verify entries, copy to output_dir.

    haddnano opens every input at once, and each ~3500-branch tree costs
    memory, so chunks with more than max_open_files inputs are merged in two
    stages through part files in staging.
    """
    output_dir = cfg["output_dir"]
    local_out = staging_dir / out_name if is_remote(output_dir) else Path(output_dir) / out_name
    local_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.unlink(missing_ok=True)

    max_open = cfg.get("max_open_files", 100)
    groups = [inputs[i : i + max_open] for i in range(0, len(inputs), max_open)]
    parts = [staging_dir / f"{out_name}.part{i}" for i in range(len(groups))] if len(groups) > 1 else []
    try:
        with open(log_dir / (out_name + ".log"), "w") as log:
            if parts:
                for part, group in zip(parts, groups):
                    run_haddnano(part, group, cfg, log)
                run_haddnano(local_out, [str(p) for p in parts], cfg, log)
            else:
                run_haddnano(local_out, inputs, cfg, log)

        with uproot.open(local_out) as f:
            got = int(f["Events"].num_entries)
        if got != expected:
            raise RuntimeError(f"{out_name}: merged {got} entries, expected {expected}")

        if is_remote(output_dir):
            copy_out(local_out, join_url(output_dir, out_name))
            # Staging holds only in-flight files; nobackup space is limited
            local_out.unlink()
    except BaseException:
        # Never leave a bad output behind: a local one would be skipped on rerun
        local_out.unlink(missing_ok=True)
        raise
    finally:
        for p in parts:
            p.unlink(missing_ok=True)
    return out_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Merge SaveEvents ROOT skims with haddnano.")
    parser.add_argument("--config", default="merge_root_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and plan, write bookkeeping locally, but do not write to output_dir.")
    parser.add_argument("--samples", nargs="*", metavar="REGEX", help="Only process dataset__sample matching any regex.")
    parser.add_argument("--force", action="store_true", help="Re-merge outputs that already exist.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, args.log_level))

    config_path = Path(args.config).resolve()
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    base_dir = config_path.parent

    def local_path(key, default):
        p = Path(cfg.get(key, default))
        return p if p.is_absolute() else base_dir / p

    source_dir = cfg["source_dir"]
    output_dir = cfg["output_dir"]
    staging_dir = local_path("staging_dir", "merge_staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    log_dir = staging_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    cfg["haddnano"] = str(local_path("haddnano", cfg["haddnano"]))

    defaults = {
        "target_size_mb": cfg.get("target_size_mb", 2000),
        "max_events": cfg.get("max_events"),
        "max_files": cfg.get("max_files", 500),
    }
    overrides = [(re.compile(o["pattern"]), o) for o in cfg.get("overrides", [])]
    selectors = [re.compile(s) for s in (args.samples or [".*"])]

    # --- list + parse --------------------------------------------------------
    logger.info("Listing %s …", source_dir)
    names_sizes = xrd_listdir(source_dir)
    logger.info("Found %d file(s).", len(names_sizes))

    parsed, unparsed = {}, []
    for n in names_sizes:
        m = FNAME_RE.match(n)
        if not m:
            unparsed.append(n)
            continue
        key = f"{m['dataset']}__{m['sample']}"
        if any(s.search(key) for s in selectors):
            parsed[n] = m
    if unparsed:
        logger.warning("Skipping %d file(s) with unrecognized names, e.g. %s", len(unparsed), unparsed[0])

    # --- inspect --------------------------------------------------------------
    info = inspect_all(
        source_dir,
        {n: names_sizes[n] for n in parsed},
        staging_dir / "inspect_cache.json",
        cfg.get("inspect_workers", 8),
    )

    samples = defaultdict(list)
    for n, m in parsed.items():
        samples[(m["dataset"], m["sample"])].append(
            {
                "name": n,
                "file_id": m["file_id"],
                "start": int(m["start"]),
                "stop": int(m["stop"]),
                "size": names_sizes[n],
                **info[n],
            }
        )

    orig_yamls, originals = load_original_yamls(cfg.get("dataset_yamls", []), base_dir)
    existing_out = {} if args.dry_run else xrd_listdir(output_dir)

    summary_rows = []
    new_values = {}
    manifest = {}
    jobs = []

    for (ds_name, s_name), files in sorted(samples.items()):
        key = f"{ds_name}__{s_name}"
        opts = dict(defaults)
        for pat, o in overrides:
            if pat.search(key):
                opts.update({k: v for k, v in o.items() if k != "pattern"})

        bad = [f for f in files if "error" in f]
        good = [f for f in files if "error" not in f]
        n_overlap = check_overlaps(good)
        n_proc = sum(f["stop"] - f["start"] for f in good)
        nonempty = sorted((f for f in good if f["entries"] > 0), key=lambda f: (f["file_id"], f["start"]))
        n_pass = sum(f["entries"] for f in nonempty)
        sign_sum = sum(f["sign_sum"] for f in nonempty)
        sumw = sum(f["sumw"] for f in nonempty)

        # x_sec_eff uses only what this skim actually processed; the yaml's
        # n_events (global total) is reported for reference only.
        orig = originals.get((ds_name, s_name))
        x_sec = orig.get("x_sec") if orig else None
        n_expected = orig.get("n_events") if orig else None
        x_sec_eff = x_sec * n_pass / n_proc if (x_sec is not None and n_proc) else None
        completeness = n_proc / n_expected if n_expected else None

        if orig is None:
            logger.warning("[%s] Not found in dataset_yamls — no x_sec available.", key)
        if bad:
            logger.warning("[%s] %d unreadable file(s) excluded from N_processed and N_pass.", key, len(bad))

        chunks = make_chunks(
            nonempty,
            int(opts["target_size_mb"] * 1024**2) if opts["target_size_mb"] else None,
            opts["max_events"],
            opts["max_files"],
        )
        out_names = [f"{key}__{i:04d}.root" for i in range(len(chunks))]

        logger.info(
            "[%s] files=%d (empty=%d)  N_proc=%d (%s of yaml n_events)  N_pass=%d  eff=%.3e  "
            "x_sec=%s → x_sec_eff=%s  → %d output(s)",
            key, len(files), len(good) - len(nonempty), n_proc,
            f"{100 * completeness:.2f}%" if completeness is not None else "?", n_pass,
            n_pass / n_proc if n_proc else 0, x_sec, f"{x_sec_eff:.6g}" if x_sec_eff is not None else None,
            len(chunks),
        )

        summary_rows.append(
            {
                "dataset": ds_name,
                "sample": s_name,
                "n_files": len(files),
                "n_empty": len(good) - len(nonempty),
                "n_unreadable": len(bad),
                "n_overlapping": n_overlap,
                "n_events_expected": n_expected,
                "n_processed": n_proc,
                "completeness": completeness,
                "n_pass": n_pass,
                "efficiency": n_pass / n_proc if n_proc else None,
                "sum_sign_genWeight_pass": sign_sum,
                "sum_genWeight_pass": sumw,
                "x_sec": x_sec,
                "x_sec_eff": x_sec_eff,
                "n_outputs": len(chunks),
            }
        )

        for chunk, out_name in zip(chunks, out_names):
            expected = sum(f["entries"] for f in chunk)
            manifest[out_name] = {"entries": expected, "inputs": [f["name"] for f in chunk]}
            if out_name in existing_out and not args.force:
                logger.debug("  %s exists, skipping", out_name)
                continue
            jobs.append(([join_url(source_dir, f["name"]) for f in chunk], out_name, expected))

        if n_pass == 0:
            logger.warning("[%s] No events passed — omitted from dataset yaml.", key)
            continue
        new_values[(ds_name, s_name)] = {
            "n_events": n_pass,
            "x_sec": float(x_sec_eff) if x_sec_eff is not None else None,
            "files": [join_url(output_dir, n) for n in out_names],
        }

    # --- write bookkeeping ------------------------------------------------------
    # A --samples run sees only part of the samples, so it must not overwrite
    # the full summary / yamls.
    meta_files = []
    if args.samples:
        logger.info("--samples given: bookkeeping not rewritten.")
    else:
        summary_path = local_path("summary", "merge_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["dataset"])
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("Summary → %s", summary_path)

        yaml_paths = write_skim_yamls(
            orig_yamls, new_values, local_path("output_yaml_dir", "merged_datasets"), cfg.get("dataset_suffix", "")
        )
        for p in yaml_paths:
            logger.info("Dataset yaml → %s", p)

        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=1))
        meta_files = [summary_path, manifest_path, *yaml_paths]

    if args.dry_run:
        logger.info("Dry run: %d merge job(s) planned, nothing merged.", len(jobs))
        return

    # --- provenance --------------------------------------------------------------
    # Copied before merging, so missing write access fails here, not per job
    meta_dir = join_url(output_dir, cfg.get("meta_subdir", "_meta"))
    try:
        if is_remote(meta_dir):
            parsed = urlparse(meta_dir)
            subprocess.run(["xrdfs", parsed.netloc, "mkdir", "-p", parsed.path], check=True)
        else:
            Path(meta_dir).mkdir(parents=True, exist_ok=True)
        for p in meta_files:
            copy_out(p, join_url(meta_dir, p.name))
    except (subprocess.CalledProcessError, OSError) as e:
        logger.error("Could not write to %s (%s) — check write access to output_dir.", meta_dir, e)
        return 1
    if meta_files:
        logger.info("Provenance → %s", meta_dir)

    # --- merge --------------------------------------------------------------------
    logger.info("Running %d merge job(s) …", len(jobs))
    failed = []
    with ThreadPoolExecutor(max_workers=cfg.get("merge_workers", 6)) as pool:
        futures = {
            pool.submit(merge_chunk, inputs, out_name, expected, cfg, staging_dir, log_dir): out_name
            for inputs, out_name, expected in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            out_name = futures[fut]
            try:
                fut.result()
                logger.info("  [%d/%d] %s", i, len(jobs), out_name)
            except Exception as e:
                failed.append(out_name)
                logger.error("  [%d/%d] %s FAILED: %s", i, len(jobs), out_name, e)

    if failed:
        logger.error("%d merge(s) failed; rerun to retry (finished outputs are skipped).", len(failed))
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    code = main()
    logging.shutdown()
    # XRootD client threads can keep the interpreter alive after main() returns,
    # which left a finished run hanging for a day; exit explicitly.
    os._exit(code or 0)
