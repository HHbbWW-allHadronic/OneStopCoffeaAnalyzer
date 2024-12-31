import argparse
import logging
import sys
from pathlib import Path
from analyzer.logging import setup_logging

logger = logging.getLogger(__name__)


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


def addSubparserCheckResult(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("check-result", help="Check result")
    subparser.add_argument("input", type=Path, help="Input data path.")
    subparser.set_defaults(func=handleCheckResult)



def handleRun(args):
    from analyzer.core.running import runFromPath
    runFromPath(args.input, args.output, args.executor, args.save_separate)



def addSubparserRun(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("run", help="Run analyzer")
    subparser.add_argument("input", type=Path, help="Input data path.")
    subparser.add_argument(
        "-o", "--output", type=Path, help="Output path", required=True
    )
    subparser.add_argument(
        "-s", "--save-separate", action="store_true", help="If set, store results separately"
    )
    subparser.add_argument(
        "-e", "--executor", type=str, help="Name of executor to use", required=True
    )
    subparser.set_defaults(func=handleRun)


def addGeneralArguments(parser):
    parser.add_argument("--log-level", type=str, default="WARN", help="Logging level")


def runCli():

    parser = argparse.ArgumentParser(prog="SingleStopAnalyzer")
    addGeneralArguments(parser)

    subparsers = parser.add_subparsers()
    addSubparserRun(subparsers)
    # addSubparserCheckResult(subparsers)
    # addSubparserGenerateReplicaCache(subparsers)

    # argcomplete.autocomplete(parser)

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
