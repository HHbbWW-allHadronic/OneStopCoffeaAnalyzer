import argparse
import logging
import sys
from pathlib import Path

from analyzer.logging import setup_logging

logger = logging.getLogger(__name__)


def makeCluster(args):
    from analyzer.clients import createNewCluster
    from distributed.client import Client

    logger.info("Handling cluster-start")
    config = {
        "max_workers": args.workers,
        "memory": args.memory,
        "schedd_host": args.scheduler,
        "dashboard_host": args.dashboard,
        "timeout": args.timeout,
    }
    client = Client(createNewCluster(args.type, config))
    return client


def handlePreprocess(args):
    from analyzer.core import preprocessAnalysis

    client = makeCluster(args)
    preprocessAnalysis(args.input, args.output)


def handlePatchPreprocess(args):
    from analyzer.core import patchPreprocessedFile

    client = makeCluster(args)
    output = args.output or args.input
    patchPreprocessedFile(args.input, output)


def handlePatchRun(args):
    from analyzer.core import patchAnalysisResult

    client = makeCluster(args)
    output = args.output or args.input
    patchAnalysisResult(args.input, output)

def handleCheckResult(args):
    from analyzer.core import checkResult

    checkResult(args.input)


def handleRun(args):
    from analyzer.core import runFromFile

    client = makeCluster(args)
    # while (client.status == "running") and (
    #     len(client.scheduler_info()["workers"]) < 20
    # ):
    #     print("Waiting")
    #     sleep(1.0)

    # with dm.memray_workers():
    runFromFile(
            args.input, args.output, preprocessed_input_path=args.preprocessed_inputs
            )
    client.shutdown(),


def handleGenReplicas(args):
    from analyzer.datasets import DatasetRepo

    dr = DatasetRepo.getConfig()
    dr.buildReplicaCache(args.force)


def commonInOut(parser):
    parser.add_argument("input", type=Path, help="Input data path.")
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output data path."
    )


def commonDask(parser):
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        help="Type of cluster",
    )
    parser.add_argument(
        "-d",
        "--dashboard",
        type=str,
        default="localhost:8787",
        help="Host for scheduler",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default=None,
        help="Host for scheduler",
    )
    parser.add_argument(
        "-w", "--workers", required=True, type=int, help="Number of workers"
    )
    parser.add_argument(
        "-m", "--memory", default="4.0G", type=str, help="Worker memory"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum time in seconds to run the cluster",
    )


def addSubparserPreprocess(subparsers):
    """Just produce preprocessed inputs"""
    subparser = subparsers.add_parser("preprocess", help="Create preprocessed inputs")
    commonInOut(subparser)
    commonDask(subparser)
    subparser.set_defaults(func=handlePreprocess)


def addSubparserPatchPreprocess(subparsers):
    """Update an existing preprocessed file with any missing info"""
    subparser = subparsers.add_parser(
        "patch-preprocessed", help="Run analyzer over some collection of samples"
    )
    commonDask(subparser)
    commonInOut(subparser)
    subparser.set_defaults(func=handlePatchPreprocess)


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


def addSubparserRun(subparsers):
    subparser = subparsers.add_parser("run-analysis", help="Run analysis")
    commonInOut(subparser)
    commonDask(subparser)
    subparser.add_argument(
        "-p",
        "--preprocessed-inputs",
        type=str,
        default=None,
        help="Preprocessed inputs",
    )
    subparser.set_defaults(func=handleRun)


def addSubparserPatchRun(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "patch-result", help="Attempt to patch issues in a results file."
    )
    commonInOut(subparser)
    commonDask(subparser)
    subparser.set_defaults(func=handlePatchRun)


def addSubparserCheckResult(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "check-result", help="Check result"
    )
    subparser.add_argument("input", type=Path, help="Input data path.")
    subparser.set_defaults(func=handleCheckResult)


def addGeneralArguments(parser):
    parser.add_argument("--log-level", type=str, default="WARN", help="Logging level")


def runCli():
    parser = argparse.ArgumentParser(prog="SingleStopAnalyzer")
    addGeneralArguments(parser)

    subparsers = parser.add_subparsers()
    addSubparserPreprocess(subparsers)
    addSubparserPatchRun(subparsers)
    addSubparserRun(subparsers)
    addSubparserCheckResult(subparsers)
    addSubparserPatchPreprocess(subparsers)
    addSubparserGenerateReplicaCache(subparsers)

    #argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)

    return args


def main():
    args = runCli()

    setup_logging(default_level=args.log_level)
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
