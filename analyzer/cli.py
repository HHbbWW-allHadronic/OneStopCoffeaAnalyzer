import argparse
import inspect
import itertools as it
import logging
import sys
from pathlib import Path

import analyzer.core as ac
import analyzer.datasets as ds
import analyzer.run_analysis as ra
from analyzer.clients import cluster_factory, createNewCluster, runNewCluster
from analyzer.core import modules as all_modules
from analyzer.plotting.simple_plot import Plotter
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    DynamicCompleter,
    NestedCompleter,
    PathCompleter,
    WordCompleter,
)
from prompt_toolkit.history import FileHistory
from rich import print
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


def handleCluster(args):
    logger.info("Handling cluster-start")
    config = {
        "n_workers": args.workers,
        "memory": args.memory,
        "schedd_host": args.scheduler,
        "dashboard_host": args.dashboard,
        "timeout": args.timeout,
    }
    client = runNewCluster(args.type, config)


def handleRunAnalysis(args):
    logger.info("Handling run analysis")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory(args.dataset_path)
    ret = ra.runAnalysisOnSamples(
        args.modules,
        args.samples,
        sample_manager,
        dask_schedd_address=args.scheduler_address,
        delayed=not args.no_delayed,
        step_size=args.step_size,
    )
    ret.save(args.output)
    if args.print_after:
        print(ret)


def handleSamples(args):
    logger.info("Handling sample inspection")
    manager = ds.SampleManager()
    manager.loadSamplesFromDirectory("datasets")

    if args.type is None:
        table = ds.createSampleAndCollectionTable(manager, re_filter=args.filter)
    elif args.type == "set":
        table = ds.createSetTable(manager, re_filter=args.filter)
    elif args.type == "collection":
        table = ds.createCollectionTable(manager, re_filter=args.filter)

    console = Console()
    console.print(table)


def handleModules(args):
    logger.info("Handling module inspection")
    import analyzer.modules

    all_modules = list(ac.modules.values())
    table = Table(
        "Name", "Categories", "Depends On", "After", "Always", title="Analysis Modules"
    )
    for module in sorted(ac.modules.values(),key=lambda x:x.name):
        table.add_row(
            module.name,
            ",".join(x for x in module.categories),
            ",".join(x for x in module.depends_on),
            ",".join(
                it.chain.from_iterable(
                    ac.org.category_after.get(y, []) for y in module.categories
                )
            ),
            str(module.always),
        )
    console = Console()
    console.print(table)


def addSubparserSamples(subparsers):
    subparser = subparsers.add_parser(
        "samples", help="Get information on available data samples"
    )
    subparser.set_defaults(func=handleSamples)
    subparser.add_argument("-f", "--filter", default=None, help="Regex to filter names")
    subparser.add_argument(
        "-t",
        "--type",
        choices=["set", "collection"],
        default=None,
        help="Show information for only samples or collections",
    )


def addSubparserModules(subparsers):
    subparser = subparsers.add_parser(
        "modules", help="Get information on available analysis modules"
    )
    subparser.set_defaults(func=handleModules)


def handleCheck(args):
    manager = ds.SampleManager()
    manager.loadSamplesFromDirectory(args.dataset_path)

    print(f"Running checks on file {args.input}")
    res = ac.AnalysisResult.fromFile(args.input)
    checks = ac.checkAnalysisResult(res, manager)

    table = Table(title="Check Result")
    table.add_column("Sample")
    table.add_column("Check Name")
    table.add_column("Passed")
    table.add_column("Info")
    print(checks)
    for sample, sample_checks in checks.items():
        for c in sample_checks:
            table.add_row(sample, c.type, str(c.passed), c.description)

    console = Console()
    console.print(table)


def addSubparserCheck(subparsers):
    subparser = subparsers.add_parser(
        "check-file", help="Identify potential problems in an output file"
    )
    subparser.add_argument("input", type=str, help="Path to the results file to check")
    subparser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path to directory containing dataset files",
    )

    subparser.set_defaults(func=handleCheck)


class ResultInspectionCli:
    def __init__(self):
        self.current_result = None
        self.dispatch = {
            "open": self.open,
            "exit": self.exit,
            "help": self.help,
        }
        self.completer = NestedCompleter.from_nested_dict(
            {
                "open": PathCompleter(),
                "exit": None,
                "help": None,
            }
        )
        self.plotter = None

        self.running = True

    def plot(self, dataset, hist, *ax_opts):
        if len(ax_opts) % 2:
            print("Incorrect ax_opts format")
            return
        a_opts = dict(
            zip(
                ax_opts[::2],
                (Plotter.Split if x == "split" else x for x in ax_opts[1::2]),
            )
        )
        figs = self.plotter(hist, [dataset], axis_opts=a_opts)

        for f in figs:
            f.show()

    def getCompleter(self):
        return self.completer

    def info(self):
        pass

    def exit(self):
        self.running = False

    def open(self, path):
        try:
            self.current_result = ac.AnalysisResult.fromFile(path)
        except IOError as e:
            print(f"Could not open {path}")
            return
        datasets = list(self.current_result.results)
        d = {
            "open": PathCompleter(),
            "help": None,
            "exit": None,
            "info": None,
            "list-contents": set(datasets),
            "show-dataset-hist": {
                name: set(res.histograms)
                for name, res in self.current_result.results.items()
            },
            "plot-hist": {
                name: set(res.histograms)
                for name, res in self.current_result.results.items()
            },
        }
        self.dispatch = {
            "open": self.open,
            "exit": self.exit,
            "info": self.info,
            "help": self.help,
            "list-contents": self.listContents,
            "show-dataset-hist": self.showHist,
            "plot-hist": self.plot,
        }
        self.completer = NestedCompleter.from_nested_dict(d)

        self.plotter = Plotter(self.current_result, None)

    def help(self):
        print(f"Available commands are:")
        print(", ".join(list(self.dispatch)))

    def showHist(self, dataset, hname):
        print(self.current_result.results[dataset].histograms[hname])

    def listContents(self):
        if not self.current_result:
            print("No file currently open.")
        print(list(self.current_result.results))

    def processString(self, string):
        import shlex

        parsed = list(shlex.split(string))
        if not parsed:
            raise Exception()
        command_name = parsed[0]
        if command_name not in self.dispatch:
            print(f'Unknown command "{command_name}"')
            self.help()
            return
        func = self.dispatch[command_name]
        try:
            func(*parsed[1:])
        except TypeError as e:
            print(e)
            print(f"Incorrect arguments passed to command {command_name}")
            return
        return

    def run(self):
        session = PromptSession(
            history=FileHistory(".inspect_cli_history"),
        )

        while self.running:
            try:
                text = session.prompt("> ", completer=self.completer)
                self.processString(text)
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

        print("Goodbye")


def handleInspect(args):
    cli = ResultInspectionCli()
    cli.run()


def addSubparserInspect(subparsers):
    subparser = subparsers.add_parser(
        "inspect", help="Inspect contents of a results file"
    )

    subparser.set_defaults(func=handleInspect)


def addSubparserCluster(subparsers):
    subparser = subparsers.add_parser("cluster-start", help="Start analysis cluster")
    subparser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        choices=list(cluster_factory),
        help="Type of cluster",
    )
    subparser.add_argument(
        "-d",
        "--dashboard",
        type=str,
        default="localhost:8787",
        help="Host for scheduler",
    )
    subparser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default=None,
        help="Host for scheduler",
    )
    subparser.add_argument("-w", "--workers", type=int, help="Number of workers")
    subparser.add_argument(
        "-m", "--memory", default="2.0G", type=str, help="Worker memory"
    )
    subparser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum time in seconds to run the cluster",
    )
    subparser.set_defaults(func=handleCluster)


def addSubparserRun(subparsers):
    subparser = subparsers.add_parser(
        "run", help="Run analyzer over some collection of samples"
    )
    subparser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output data path."
    )

    run_mode = subparser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "-a",
        "--scheduler-address",
        type=str,
        help="Address of the scheduler to use for dask",
    )

    run_mode.add_argument(
        "--no-delayed",
        action="store_true",
        default=False,
        help="Do not use dask, instead run synchronously. Good for testing.",
    )

    subparser.add_argument(
        "-s",
        "--samples",
        type=str,
        nargs="+",
        help="List of samples to run over",
        metavar="",
    )
    subparser.add_argument(
        "-m",
        "--modules",
        type=str,
        nargs="+",
        help="List of modules to execute.",
        metavar="",
    )

    subparser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path to directory containing dataset files",
    )

    subparser.add_argument(
        "--save-graph",
        default=None,
        type=str,
        help="In set, interpret as a path to which the task-graph should be saved.",
    )

    subparser.add_argument(
        "--print-after",
        default=False,
        action="store_true",
        help="If true, print the result.",
    )

    subparser.add_argument(
        "--step-size",
        default=100000,
        type=int,
        help="Number of events per chunk",
    )

    subparser.set_defaults(func=handleRunAnalysis)


def addGeneralArguments(parser):
    parser.add_argument("--log-level", type=str, default="WARN", help="Logging level")


def runCli():
    parser = argparse.ArgumentParser(prog="SingleStopAnalyzer")
    addGeneralArguments(parser)

    subparsers = parser.add_subparsers()
    addSubparserCluster(subparsers)
    addSubparserRun(subparsers)
    addSubparserSamples(subparsers)
    addSubparserCheck(subparsers)
    addSubparserModules(subparsers)
    addSubparserInspect(subparsers)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)

    return args
