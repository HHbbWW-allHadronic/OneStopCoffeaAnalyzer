import argparse
import inspect
import itertools as it
import sys
import pickle as pkl
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    DynamicCompleter,
    NestedCompleter,
    PathCompleter,
    WordCompleter,
)
from prompt_toolkit.history import FileHistory
import argcomplete
from .configuration import getConfiguration

from .logging import setup_logging
import logging

logger = logging.getLogger(__name__)


def saveCachedArgparseList(name, data):
    from analyzer.file_utils import pickleWithParents

    config_path = Path(getConfiguration()["APPLICATION_DATA"])
    path = config_path / "argparse_cache" / f"{name}.pkl"
    pickleWithParents(path, data)


def loadCachedArgparseList(name):
    config_path = Path(getConfiguration()["APPLICATION_DATA"])
    path = config_path / "argparse_cache" / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pkl.load(f)
    else:
        return None


def getSampleRepos(args):
    import analyzer.datasets as ds

    profile_repo = ds.ProfileRepo()
    profile_repo.loadFromDirectory("profiles")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory(
        args.dataset_path, profile_repo, use_replicas=args.use_replicas
    )
    saveCachedArgparseList(
        "samples", list(it.chain(sample_manager.sets, sample_manager.collections))
    )
    return sample_manager, profile_repo


def handleCluster(args):
    from analyzer.clients import cluster_factory, createNewCluster, runNewCluster

    logger.info("Handling cluster-start")
    config = {
        "n_workers": args.workers,
        "memory": args.memory,
        "schedd_host": args.scheduler,
        "dashboard_host": args.dashboard,
        "timeout": args.timeout,
    }
    client = runNewCluster(args.type, config)


def handlePreprocess(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents

    logger.info("Handling preprocess")

    sample_manager, profile_repo = getSampleRepos(args)
    client = ra.createClient(args.scheduler_address, not args.no_transfer_analyzer)
    prepped = ra.createPreprocessedSamples(
        sample_manager,
        args.samples,
        args.step_size,
        file_retrieval_kwargs=dict(
            location_priority_regex=[r".*(US|CH|FR).*", "eos"],
            require_location=args.require_location,
        ),
    )
    pickleWithParents(args.output, prepped)


def handleCheckPreprocess(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents

    logger.info("Handling preprocess")

    sample_manager, profile_repo = getSampleRepos(args)
    with open(args.input, "rb") as f:
        prepped = pkl.load(f)

    for dataset in prepped:
        print(dataset.dataset_name, dataset.missingFiles())


def handlePatchPreprocess(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents

    sample_manager, profile_repo = getSampleRepos(args)
    client = ra.createClient(args.scheduler_address, not args.no_transfer_analyzer)

    with open(args.input, "rb") as f:
        prepped = pkl.load(f)

    patched = ra.patchPreprocessed(sample_manager, prepped, step_size=args.step_size)

    pickleWithParents(args.output, patched)


def handleRunPreprocessed(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents

    logger.info("Handling run-preprocessed")
    sample_manager, profile_repo = getSampleRepos(args)
    client = ra.createClient(args.scheduler_address, not args.no_transfer_analyzer)

    with open(args.preprocessed_inputs, "rb") as f:
        prepped = pkl.load(f)
    result = ra.runModulesOnDatasets(
        args.modules,
        prepped,
        client,
        args.skim_save_path,
        file_retrieval_kwargs=dict(
            location_priority_regex=[r".*(US|CH|FR).*", "eos"],
            require_location=args.require_location,
        ),
    )

    pickleWithParents(args.output, result)


def handleGenReplicas(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents

    profile_repo = ds.ProfileRepo()
    profile_repo.loadFromDirectory("profiles")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory(args.dataset_path, profile_repo)

    sample_manager.buildReplicaCache(args.force)


def handleRunSamples(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents

    sample_manager, profile_repo = getSampleRepos(args)
    client = ra.createClient(args.scheduler_address, not args.no_transfer_analyzer)

    prepped = ra.createPreprocessedSamples(
        sample_manager,
        args.samples,
        args.step_size,
        file_retrieval_kwargs=dict(
            location_priority_regex=[r".*(US|CH|FR).*", "eos"],
            require_location=args.require_location,
        ),
    )
    if args.save_preprocessed:
        pickleWithParents(args.save_preprocessed, prepped)
    result = ra.runModulesOnDatasets(
        args.modules,
        prepped,
        client=client,
        skim_save_path=args.skim_save_path,
        file_retrieval_kwargs=dict(
            location_priority_regex=[r".*(US|CH|FR).*", "eos"],
            require_location=args.require_location,
        ),
    )
    pickleWithParents(args.output, result)


def handlePatchRun(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents

    sample_manager, profile_repo = getSampleRepos(args)
    client = ra.createClient(args.scheduler_address, not args.no_transfer_analyzer)

    with open(args.input, "rb") as f:
        results = pkl.load(f)

    patched = ra.patchResult(results, client=client, skim_save_path=args.skim_save_path)

    pickleWithParents(args.output, patched)


def handleSamples(args):
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents
    from rich import print
    from rich.columns import Columns
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import Pretty
    from rich.table import Table

    logger.info("Handling sample inspection")

    manager, profile_repo = getSampleRepos(args)

    if args.type is None:
        table = ds.createSampleAndCollectionTable(manager, re_filter=args.filter)
    elif args.type == "set":
        table = ds.createSetTable(manager, re_filter=args.filter)
    elif args.type == "collection":
        table = ds.createCollectionTable(manager, re_filter=args.filter)

    console = Console()
    console.print(table)


def handleModules(args):
    import analyzer.core as ac
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from analyzer.file_utils import pickleWithParents
    import analyzer.modules
    from rich import print
    from rich.columns import Columns
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import Pretty
    from rich.table import Table

    console = Console()

    if args.modules:
        for m in args.modules:

            try:
                module = ac.modules[m]
            except KeyError as e:
                print(f"No module {m}")
                continue
            text = f"""
# Module: {module.name}
{module.documenation or ''}"""
            console.print(Markdown(text))
            used_fields, created_fields, hists = ac.org.inspectModule(module)
            if len(used_fields):
                used_text = "## Used Fields"
                columns = Columns(sorted(used_fields), equal=True, expand=True)
                console.print(Markdown(used_text))
                console.print(columns)
            if len(created_fields):
                created_text = "## Created Fields"
                columns = Columns(sorted(created_fields), equal=True, expand=True)
                console.print(Markdown(created_text))
                console.print(columns)
            if len(hists):
                h_text = "## Created Histograms"
                columns = Columns(sorted(hists), equal=True, expand=True)
                console.print(Markdown(h_text))
                console.print(columns)

    else:
        all_modules = list(ac.modules.values())
        table = Table(
            "Name",
            "Categories",
            "Depends On",
            "After",
            "Always",
            title="Analysis Modules",
        )
        for module in sorted(ac.modules.values(), key=lambda x: x.name):
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
        console.print(table)


def addSubparserSamples(subparsers):
    subparser = subparsers.add_parser(
        "samples", help="Get information on available data samples"
    )
    subparser = addCommonPathArgs(subparser)

    subparser.add_argument("-f", "--filter", default=None, help="Regex to filter names")
    subparser.add_argument(
        "-t",
        "--type",
        choices=["set", "collection"],
        default=None,
        help="Show information for only samples or collections",
    )

    subparser.set_defaults(func=handleSamples)


def addSubparserModules(subparsers):
    subparser = subparsers.add_parser(
        "modules", help="Get information on available analysis modules"
    )
    subparser.set_defaults(func=handleModules)
    subparser.add_argument(
        "modules",
        nargs="*",
        help="If provided, give verbose information on listed modules.",
    )


def handleCheck(args):

    import analyzer.core as ac
    import analyzer.datasets as ds
    import analyzer.run_analysis as ra
    from rich import print
    from rich.columns import Columns
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import Pretty
    from rich.table import Table

    profile_repo = ds.ProfileRepo()
    profile_repo.loadFromDirectory("profiles")
    manager = ds.SampleManager()
    manager.loadSamplesFromDirectory(args.dataset_path, profile_repo)

    print(f"Running checks on file {args.input}")
    res = ac.AnalysisResult.fromFile(args.input)
    checks = ac.checkAnalysisResult(res, manager)

    table = Table(title="Check Result")
    table.add_column("Sample")
    table.add_column("Check Name")
    table.add_column("Passed")
    table.add_column("Info")
    # print(checks)
    for sample, sample_checks in checks.items():
        for c in sample_checks:
            if not args.only_bad or not c.passed:
                table.add_row(sample, c.type, str(c.passed), c.description)

    console = Console()
    console.print(table)
    if all(x.passed for x in it.chain.from_iterable(checks.values())):
        print("ALL CHECKS HAVE PASSED")
    else:
        print("THERE WERE FAILED CHECKS, YOU LIKELY WANT TO PATCH THIS RESULTS FILE")


def addSubparserCheck(subparsers):
    subparser = subparsers.add_parser(
        "check-result", help="Identify potential problems in a results file"
    )
    subparser.add_argument("input", type=str, help="Path to the results file to check")
    subparser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path to directory containing dataset files",
    )

    subparser.add_argument(
        "--only-bad",
        default=False,
        action="store_true",
        help="Only show problems",
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
        # choices=list(cluster_factory),
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


def addCommonPathArgs(parser):
    parser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path to directory containing dataset files",
    )

    parser.add_argument(
        "--profile-path",
        default="profiles",
        type=str,
        help="Path to directory containing profile files",
    )
    parser.add_argument(
        "--use-replicas",
        type=bool,
        default=True,
        help="Use replicas for samples",
    )
    return parser


def addCommonDaskArgs(parser):
    parser.add_argument(
        "--no-transfer-analyzer",
        default=False,
        action="store_true",
        help="If set, do not transfer the analyzer. Useful if rerunning over an existing cluster when the analyzer has not changed.",
    )

    parser.add_argument(
        "-a",
        "--scheduler-address",
        type=str,
        help="Address of the scheduler to use for dask",
    )

    return parser


def addCommonPrepArgs(parser):
    sample_choices = loadCachedArgparseList("samples")
    if not sample_choices:
        import analyzer.datasets as ds
        import analyzer.run_analysis as ra
        from analyzer.file_utils import pickleWithParents

        profile_repo = ds.ProfileRepo()
        profile_repo.loadFromDirectory("profiles")
        s = ds.SampleManager()
        s.loadSamplesFromDirectory(
            "datasets",
            profile_repo,
            use_replicas=False,
        )
        saveCachedArgparseList("samples", list(it.chain(s.sets, s.collections)))

    parser.add_argument(
        "--step-size",
        default=100000,
        type=int,
        help="Number of events per chunk",
    )

    parser.add_argument(
        "--require-location",
        default=None,
        type=str,
        help="If provided, require that all samples be found at provided location.",
    )

    parser.add_argument(
        "--prefer-location",
        default="eos",
        type=str,
        help="If provided, prefer that all samples be found at provided location.",
    )

    parser.add_argument(
        "-s",
        "--samples",
        type=str,
        nargs="+",
        help="List of samples to run over",
        metavar="",
        choices=sample_choices,
    )

    return parser


def addSubparserPreprocess(subparsers):
    """Just produce preprocessed inputs"""
    subparser = subparsers.add_parser("preprocess", help="Create preprocessed inputs")
    subparser = addCommonPathArgs(subparser)
    subparser = addCommonDaskArgs(subparser)
    subparser = addCommonPrepArgs(subparser)

    subparser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output data path."
    )
    subparser.set_defaults(func=handlePreprocess)


def addCommonRunArgs(parser):
    module_choices = loadCachedArgparseList("modules")
    if not module_choices:
        import analyzer.modules

        saveCachedArgparseList("modules", list(analyzer.core.org.modules))

    parser.add_argument(
        "-m",
        "--modules",
        type=str,
        nargs="+",
        help="List of modules to execute.",
        choices=module_choices,
        metavar="",
    )

    parser.add_argument(
        "--save-graph",
        default=None,
        type=str,
        help="In set, interpret as a path to which the task-graph should be saved.",
    )

    parser.add_argument(
        "--skim-save-path",
        type=str,
        default=None,
        help="Path to use to save skims.",
    )

    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output data path."
    )
    return parser


def addSubparserRunPreprocessed(subparsers):
    """Run over a preprocessed fileset"""
    subparser = subparsers.add_parser(
        "run-preprocessed", help="Run analyzer over preprocessed inputs"
    )
    subparser = addCommonPathArgs(subparser)
    subparser = addCommonDaskArgs(subparser)
    subparser = addCommonRunArgs(subparser)

    subparser.add_argument(
        "--preprocessed-inputs",
        default=None,
        required=True,
        type=str,
        help="The path to a the preprocessed dataset",
    )
    subparser.set_defaults(func=handleRunPreprocessed)


def addSubparserRunSamples(subparsers):
    """Run over samples"""

    subparser = subparsers.add_parser(
        "run-samples", help="Run analyzer over some collection of samples"
    )

    subparser = addCommonPathArgs(subparser)
    subparser = addCommonDaskArgs(subparser)
    subparser = addCommonPrepArgs(subparser)
    subparser = addCommonRunArgs(subparser)

    subparser.add_argument(
        "--save-preprocessed",
        default=None,
        type=str,
        help="If provided, the path to a file to save preprocessed datasets to. This data may later be used as input to the analyzer.",
    )

    subparser.set_defaults(func=handleRunSamples)


def addSubparserPatchPreprocess(subparsers):
    """Update an existing preprocessed file with any missing info"""
    subparser = subparsers.add_parser(
        "patch-preprocessed", help="Run analyzer over some collection of samples"
    )

    subparser = addCommonPathArgs(subparser)
    subparser = addCommonDaskArgs(subparser)

    subparser.add_argument(
        "--step-size",
        default=150000,
        type=int,
        help="Number of events per chunk",
    )

    subparser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output data path."
    )
    subparser.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="Input file",
    )
    subparser.set_defaults(func=handlePatchPreprocess)


def addSubparserCheckPreprocess(subparsers):
    """Update an existing preprocessed file with any missing info"""
    subparser = subparsers.add_parser(
        "check-preprocessed", help="Run analyzer over some collection of samples"
    )

    subparser = addCommonPathArgs(subparser)

    subparser.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="Input file",
    )

    subparser.set_defaults(func=handleCheckPreprocess)


def addSubparserGenerateReplicaCache(subparsers):

    subparser = subparsers.add_parser(
        "generate-replicas",
        help="Use Rucio to get the replicas for datasets representing a standard CMS sample",
    )

    subparser = addCommonPathArgs(subparser)
    subparser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Regenerate already existing replicas as well.",
    )
    subparser.set_defaults(func=handleGenReplicas)


def addSubparserPatchRun(subparsers):
    """Update an existing results file with missing info"""
    subparser = subparsers.add_parser(
        "patch-result", help="Attempt to patch issues in a results file."
    )
    subparser = addCommonPathArgs(subparser)
    subparser = addCommonDaskArgs(subparser)
    subparser.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="Input file",
    )
    subparser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output data path."
    )
    subparser.add_argument(
        "--skim-save-path",
        type=str,
        default=None,
        help="Path to use to save skims.",
    )
    subparser.set_defaults(func=handlePatchRun)


def addGeneralArguments(parser):
    parser.add_argument("--log-level", type=str, default="WARN", help="Logging level")


def runCli():
    parser = argparse.ArgumentParser(prog="SingleStopAnalyzer")
    addGeneralArguments(parser)

    subparsers = parser.add_subparsers()
    addSubparserCluster(subparsers)
    addSubparserRunPreprocessed(subparsers)
    addSubparserRunSamples(subparsers)
    addSubparserPreprocess(subparsers)
    addSubparserPatchRun(subparsers)
    addSubparserPatchPreprocess(subparsers)
    addSubparserCheckPreprocess(subparsers)
    addSubparserGenerateReplicaCache(subparsers)

    addSubparserSamples(subparsers)
    addSubparserCheck(subparsers)
    addSubparserModules(subparsers)
    addSubparserInspect(subparsers)

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
