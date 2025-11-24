import logging
from rich import print, get_console
import click
from enum import Enum

logger = logging.getLogger(__name__)

def jumpIn(**kwargs):
    import code
    import readline
    import rlcompleter

    vars = globals()
    vars.update(locals())
    vars.update(kwargs)
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    code.InteractiveConsole(vars).interact()


class LogLevel(Enum):
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


@click.group()
@click.option(
    "--log-level",
    default=LogLevel.WARNING,
    type=click.Choice(LogLevel, case_sensitive=False),
)
def cli(log_level):
    pass


def checkResults():
    pass


# def handleCheckResults(args):
#     from analyzer.core.results import checkResult
#
#     checkResult(args.input, configuration=args.configuration, only_bad=args.only_bad)
#
#
# def addSubparserCheckResult(subparsers):
#     """Update an existing results file with missing info"""
#     subparser = subparsers.add_parser("check-results", help="Check results")
#     subparser.add_argument("input", nargs="+", type=Path, help="Input data paths.")
#     subparser.add_argument(
#         "-b",
#         "--only-bad",
#         action="store_true",
#         default=False,
#         help="Only show samples with potential problems",
#     )
#     subparser.add_argument(
#         "-c",
#         "--configuration",
#         type=str,
#         default=None,
#         help="Optionally provide a configuration to check for completely omitted samples",
#     )
#     subparser.set_defaults(func=handleCheckResults)

# def quickEvents(
#     name, sample_name=None, nevents=10000, tree_name="Events", delayed=False
# ):
#     from analyzer.datasets import DatasetRepo, EraRepo
#     import awkward as ak
#     from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
#     import random
#
#     repo = DatasetRepo.getConfig()
#     era_repo = EraRepo.getConfig()
#     ds = None
#     try:
#         ds = repo[name]
#     except KeyError as e:
#         print(f"Could not find dataset '{name}', treating as file path")
#     if ds is not None:
#         try:
#             sname = sample_name or name
#             sample = ds[sname]
#             sample.useFilesFromReplicaCache()
#         except KeyError as e:
#             print(
#                 f"Could not find sample '{sname}'. Available samples are {[x.name for x in ds]}"
#             )
#             return
#         fname = random.choice(sample.files).getFile()
#     else:
#         fname = name
#
#     print(f"Loading events from file {fname}...")
#     events = NanoEventsFactory.from_root(
#         {fname: tree_name},
#         schemaclass=NanoAODSchema,
#         entry_start=0,
#         entry_stop=nevents,
#         delayed=delayed,
#     ).events()
#
#     return events
#
#
# def handleQuickEvents(args):
#     from analyzer.utils.debugging import jumpIn
#
#     events = quickEvents(
#         args.dataset_name, args.sample_name, args.nevents, args.tree_name
#     )
#
#     if events is not None:
#         jumpIn(events=events)


# def addSubparserQuickEvents(subparsers):
#     """Update an existing results file with missing info"""
#     subparser = subparsers.add_parser(
#         "quick-events", help="Construct datasets from simple descriptions."
#     )
#     subparser.add_argument("dataset_name")
#     subparser.add_argument("sample_name", nargs="?")
#     subparser.add_argument(
#         "-n",
#         "--nevents",
#         default=10000,
#         type=int,
#     )
#     subparser.add_argument("-t", "--tree-name", type=str, default="Events")
#     subparser.set_defaults(func=handleQuickEvents)

# def handleSamples(args):
#     from .sample_report import createSampleTable, createDatasetTable
#     from analyzer.datasets import DatasetRepo, EraRepo
#     from analyzer.utils.querying import pattern_expr_adapter, MultiPatternExpression
#
#     if args.filter:
#         filter_pattern = MultiPatternExpression(exprs=args.filter, op="AND")
#     else:
#         filter_pattern = None
#     repo = DatasetRepo.getConfig()
#     era_repo = EraRepo.getConfig()
#     repo.populateEras(era_repo)
#     if args.dataset_only:
#         table = createDatasetTable(repo, pattern=filter_pattern, as_csv=args.csv)
#     else:
#         table = createSampleTable(repo, pattern=filter_pattern,as_csv=args.csv)
#     print(table)
#
#
# def addSubparserSampleReport(subparsers):
#     from analyzer.utils.querying import pattern_expr_adapter
#     import json
#
#     subparser = subparsers.add_parser(
#         "samples", help="Get information on available samples"
#     )
#
#     def keyValuePattern(p):
#         k, v = p.split("=")
#         return pattern_expr_adapter.validate_python({k: v})
#
#     subparser.add_argument(
#         "-d",
#         "--dataset-only",
#         action="store_true",
#         default=False,
#     )
#
#     subparser.add_argument(
#         "--csv",
#         action="store_true",
#         default=False,
#     )
#
#     subparser.add_argument(
#         "--filter",
#         nargs="+",
#         type=keyValuePattern,
#         required=False,
#         default=None,
#     )
#     subparser.set_defaults(func=handleSamples)
#
#
# def handleSummaryTable(args):
#     from analyzer.tools.summary_table import (
#         createEraTable,
#         makeTableFromDict,
#         texTable,
#     )
#     from analyzer.datasets import EraRepo
#     from analyzer.utils.querying import PatternExpression, pattern_expr_adapter
#
#     query = pattern_expr_adapter.validate_python({x: "*" for x in args.fields})
#     format_opts = {"TT": "\\texttt{{{}}}", "RM": "\\textrm{{{}}}"}
#
#     era_repo = EraRepo.getConfig()
#     r = createEraTable(
#         era_repo,
#         query,
#     )
#     t = makeTableFromDict(r)
#     format_funcs = {
#         i: lambda x: format_opts[k].format(x) for i, k in args.extra_format or []
#     }
#     print(texTable(t, col_format_funcs=format_funcs))
#
#
# def addSubparserSummaryTable(subparsers):
#     """Update an existing results file with missing info"""
#     subparser = subparsers.add_parser("summary-table", help="Get summary table")
#     subparser.add_argument("-f", "--fields", type=str, nargs="+")
#
#     def formatPair(arg):
#         sp = arg.split(",")
#         return (int(sp[0]), str(sp[1]))
#
#     subparser.add_argument("-e", "--extra-format", type=formatPair, nargs="*")
#     subparser.set_defaults(func=handleSummaryTable)


@cli.command()
@click.argument("input", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--executor", "-e", type=str, required=True)
def run(
    input,
    output,
    executor,
):
    from analyzer.core.running import runFromPath

    runFromPath(input, output, executor)


@cli.command()
@click.argument("input", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--executor", "-e", type=str, required=True)
def patch(
    input,
    output,
    executor,
):
    print(output)

    pass


@cli.command()
@click.argument("inputs", type=str, nargs=-1)
@click.option("--interpretter", is_flag=True)
def browse(inputs, interpretter):
    from analyzer.core.results import loadResults
    from analyzer.core.serialization import setupConverter, converter
    setupConverter(converter)

    print(f"Loading Results")
    if interpretter:
        res = loadResults(inputs)
        jumpIn(results=res)

@cli.group()
def cache():
    pass

@cache.command()
def clear():
    from analyzer.core.caching import cache
    cache.clear()



@cli.group()
def listData():
    pass


@listData.group()
def samples():
    pass


@listData.group()
def datasets():
    pass


@listData.group()
def eras():
    pass


def main():
    cli()
