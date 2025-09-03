# PYTHON_ARGCOMPLETE_OK
import argparse
import logging
import sys
from pathlib import Path

from analyzer.logging import setup_logging
from rich import print

logger = logging.getLogger(__name__)


def handleGenReplicas(args):
    from analyzer.datasets import DatasetRepo

    dr = DatasetRepo.getConfig()
    dr.buildReplicaCache(args.force)


def addSubparserGenerateReplicaCache(subparsers):
    subparser = subparsers.add_parser(
        "generate-replicas",
        help="Use Rucio to get the replicas for datasets representing a standard CMS sample",
    )
    subparser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Regenerate already existing replicas as well.",
    )
    subparser.set_defaults(func=handleGenReplicas)


def handleUpdateMeta(args):
    from analyzer.core.results import updateMeta

    updateMeta(args.input)


def handleMerge(args):
    from analyzer.core.results import merge

    merge(args.input, args.outdir)


def addSubparserUpdateMetaInfo(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "update-meta", help="Update Metadata based on new configuration options"
    )
    subparser.add_argument("input", nargs="+", help="Input file")
    subparser.set_defaults(func=handleUpdateMeta)


def handleCheckResults(args):
    from analyzer.core.results import checkResult

    checkResult(args.input, configuration=args.configuration)


def addSubparserCheckResult(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("check-results", help="Check results")
    subparser.add_argument("input", nargs="+", type=Path, help="Input data paths.")
    subparser.add_argument(
        "-c",
        "--configuration",
        type=str,
        default=None,
        help="Optionally provide a configuration to check for completely omitted samples",
    )
    subparser.set_defaults(func=handleCheckResults)


def handleQuickDataset(args):
    from analyzer.tools.quick_dataset import run

    run(args.input, args.output_dir, args.limit_regex)


def addSubparserQuickDataset(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "dataset-builder", help="Construct datasets from simple descriptions."
    )
    subparser.add_argument("input", type=Path, help="Input data path.")
    subparser.add_argument(
        "-o",
        "--output-dir",
        required=True,
    )
    subparser.add_argument("-l", "--limit-regex")
    subparser.set_defaults(func=handleQuickDataset)


def handleStoreResults(args):
    from analyzer.tools.save_results import storeResults

    storeResults(args.input, f"{args.prefix}_{args.output_name}")


def addSubParserStoreResults(subparsers):
    """Update an existing results file with missing info"""
    import os
    import datetime

    subparser = subparsers.add_parser("store-results", help="store results")
    subparser.add_argument("-o", "--output-name")
    subparser.add_argument(
        "--eos-path",
        type=str,
        required=False,
        default=f"/store/user/{os.getlogin()}/single_stop/results/",
    )
    subparser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default=datetime.datetime.now().strftime("%Y-%m-%d"),
    )
    subparser.add_argument("input")
    subparser.set_defaults(func=handleStoreResults)


def handleSamples(args):
    from .sample_report import createSampleTable
    from analyzer.datasets import DatasetRepo
    from rich.console import Console

    console = Console()
    repo = DatasetRepo.getConfig()
    table = createSampleTable(repo)
    console.print(table)


def addSubparserSampleReport(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "samples", help="Get information on available samples"
    )
    subparser.set_defaults(func=handleSamples)


def handleSummaryTable(args):
    from analyzer.tools.summary_table import (
        createEraTable,
        makeTableFromDict,
        texTable,
    )
    from analyzer.datasets import EraRepo
    from analyzer.utils.querying import PatternExpression, pattern_expr_adapter

    query = pattern_expr_adapter.validate_python({x: "*" for x in args.fields})
    format_opts = {"TT": "\\texttt{{{}}}", "RM": "\\textrm{{{}}}"}

    era_repo = EraRepo.getConfig()
    r = createEraTable(
        era_repo,
        query,
    )
    t = makeTableFromDict(r)
    format_funcs = {
        i: lambda x: format_opts[k].format(x) for i, k in args.extra_format or []
    }
    print(texTable(t, col_format_funcs=format_funcs))


def addSubparserSummaryTable(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("summary-table", help="Get summary table")
    subparser.add_argument("-f", "--fields", type=str, nargs="+")

    def formatPair(arg):
        sp = arg.split(",")
        return (int(sp[0]), str(sp[1]))

    subparser.add_argument("-e", "--extra-format", type=formatPair, nargs="*")
    subparser.set_defaults(func=handleSummaryTable)


def handleRunPackaged(args):
    from analyzer.core.executors import PackagedTask

    with open(args.input, "rb") as f:
        task = f.load(f)
        task = PackagedTask(**task)
    runPackagedTask(task, output_dir=args.output_dir)


def addSubparserRunPackaged(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("run-packaged", help="Run a packaged task")
    subparser.add_argument("input", type=Path, help="Input data path.")
    subparser.add_argument("--output-dir", type=Path, help="Output data path.")
    subparser.set_defaults(func=handleRunPackaged)


def handleRun(args):
    from analyzer.core.running import runFromPath

    runFromPath(
        args.input,
        args.output,
        args.executor,
        test_mode=args.test_mode,
        filter_samples=args.filter_samples,
    )


def handleStartCluster(args):
    from analyzer.core.executors import LPCCondorDask
    import time

    print("HELLO")
    cluster = LPCCondorDask(max_workers=2)
    cluster.setup()
    print(cluster._cluster)
    time.sleep(20000)


def addSubparserRun(subparsers):
    """Update an existing results file with missing info"""

    def parsePatterns(p):
        from analyzer.utils.querying import Pattern

        return Pattern(pattern=p)

    subparser = subparsers.add_parser(
        "run", help="Run analyzer based on provided configuration"
    )
    subparser.add_argument("input", type=Path, help="Input data path.")
    subparser.add_argument(
        "-o", "--output", type=Path, help="Output path", required=True
    )
    subparser.add_argument(
        "--test-mode",
        default=False,
        action="store_true",
        help="Run in test mode",
    )
    subparser.add_argument(
        "--filter-samples",
        nargs="*",
        type=parsePatterns,
        required=False,
        default=None,
        help="Filter samples",
    )
    subparser.add_argument(
        "-s",
        "--save-separate",
        action="store_true",
        help="If set, store results separately",
    )
    subparser.add_argument(
        "-e", "--executor", type=str, help="Name of executor to use", required=True
    )
    subparser.set_defaults(func=handleRun)


def addSubparserStartCluster(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "start-cluster", help="Run analyzer based on provided configuration"
    )
    subparser.set_defaults(func=handleStartCluster)


def handlePost(args):
    from analyzer.postprocessing.running import runPostprocessors

    return runPostprocessors(args.config, args.input, parallel=args.parallel)


def addSubparserPost(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("postprocess", help="Postprocessing utilities")
    subparser.add_argument("input", nargs="+", type=Path, help="Input files path.")
    subparser.add_argument(
        "-c", "--config", type=Path, help="Configuration path", required=True
    )
    subparser.add_argument(
        "-p", "--parallel", type=int, default=None, help="Number of processes to use"
    )
    subparser.set_defaults(func=handlePost)


def handleQuicklook(args):
    from .quicklook import quicklookFiles, quicklookHistsPath

    if args.region_name:
        quicklookHistsPath(args.input, args.region_name, args.hist_name, args.interact)
    else:
        quicklookFiles(args.input)


def addSubparserQuicklookFile(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "quicklook", help="Quick information about a result."
    )
    subparser.add_argument("input", nargs="+", type=Path, help="Input files path.")
    subparser.add_argument(
        "-r", "--region-name", default=None, type=str, help="Region name"
    )
    subparser.add_argument(
        "-n", "--hist-name", default=None, type=str, help="hist_name"
    )
    subparser.add_argument(
        "-i", "--interact", action="store_true", default=False, help="interact"
    )
    subparser.set_defaults(func=handleQuicklook)


def handlePatch(args):
    from analyzer.core.running import patchFromPath

    patchFromPath(
        args.input,
        args.output,
        args.executor,
        args.description,
        args.ignore_ret_prefs,
    )


def addSubparserPatch(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "patch", help="Patch a result by running over missing chunks."
    )
    subparser.add_argument("input", nargs="+", type=Path, help="Input data path.")
    subparser.add_argument(
        "-o", "--output", type=Path, help="Output path", required=True
    )
    subparser.add_argument(
        "-s",
        "--save-separate",
        action="store_true",
        help="If set, store results separately",
    )
    subparser.add_argument(
        "-e", "--executor", type=str, help="Name of executor to use", required=True
    )
    subparser.add_argument(
        "-i", "--ignore-ret-prefs", action="store_true", default=False
    )

    subparser.add_argument(
        "-d", "--description", type=str, help="Description", required=True
    )
    subparser.set_defaults(func=handlePatch)


def addSubparserMerge(subparsers):
    subparser = subparsers.add_parser("merge", help="Merge multiple output files")
    subparser.add_argument("input", nargs="+", type=Path, help="Input data path.")
    subparser.add_argument(
        "-o", "--outdir", type=Path, help="Output path", required=True
    )
    subparser.set_defaults(func=handleMerge)


def addGeneralArguments(parser):
    parser.add_argument("--log-level", type=str, default="WARN", help="Logging level")


def runCli():
    import argcomplete

    parser = argparse.ArgumentParser(prog="SingleStopAnalyzer")
    addGeneralArguments(parser)

    subparsers = parser.add_subparsers()
    addSubparserRun(subparsers)
    addSubparserSampleReport(subparsers)
    addSubparserPatch(subparsers)
    addSubparserCheckResult(subparsers)
    addSubparserGenerateReplicaCache(subparsers)
    addSubparserQuicklookFile(subparsers)
    addSubparserPost(subparsers)
    addSubparserQuickDataset(subparsers)
    addSubParserStoreResults(subparsers)
    addSubparserSummaryTable(subparsers)
    addSubparserStartCluster(subparsers)
    addSubparserUpdateMetaInfo(subparsers)
    addSubparserMerge(subparsers)

    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)

    return args


def main():
    args = runCli()

    setup_logging(default_level=args.log_level)
    if hasattr(args, "func"):
        args.func(args)
