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


def handleCheckResults(args):
    from analyzer.core.results import checkResult
    checkResult(args.input)

def addSubparserCheckResult(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("check-results", help="Check result")
    subparser.add_argument("input", nargs="+", type=Path, help="Input data path.")
    subparser.set_defaults(func=handleCheckResults)



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
    subparser = subparsers.add_parser("samples", help="Run analyzer")
    subparser.set_defaults(func=handleSamples)


def handleRunPackaged(args):
    from analyzer.core.running import runFromPath
    from analyzer.core.executor import PackagedTask
    import analyzer.modules
    import pickle as pkl
    with open(args.input, 'rb') as  f:
        task = f.load(f)
        task = PackagedTask(**task)
    runPackagedTask(task, output_dir = args.output_dir)




def addSubparserRunPackaged(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser("run-packaged", help="Run analyzer")
    subparser.add_argument("input", type=Path, help="Input data path.")
    subparser.add_argument("--output-dir", type=Path, help="Output data path.")
    subparser.set_defaults(func=handleRunPackaged)


def handleRun(args):
    from analyzer.core.running import runFromPath
    import analyzer.modules
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
    addSubparserSampleReport(subparsers)
    addSubparserCheckResult(subparsers)
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


