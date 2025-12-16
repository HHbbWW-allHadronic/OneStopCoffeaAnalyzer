import logging
from rich import print, get_console
import click
from enum import Enum

logger = logging.getLogger("analyzer")


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


class LogLevel(str, Enum):
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


@cli.command()
@click.argument("input", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--executor", "-e", type=str, required=True)
@click.option("--max-sample-events", type=int, default=None)
def run(
    input,
    output,
    executor,
    max_sample_events,
):
    from analyzer.core.running import runFromPath

    runFromPath(input, output, executor, max_sample_events=max_sample_events)


@cli.command()
@click.argument("files", nargs=-1)
@click.option("--configuration", "-c", type=str, required=str)
@click.option("--only-bad", is_flag=True)
def check(files, configuration, only_bad):
    from analyzer.core.results import checkResults
    from analyzer.core.datasets import DatasetRepo
    from analyzer.core.era import EraRepo
    from analyzer.core.analysis import loadAnalysis, getSamples
    from analyzer.core.running import getRepos
    from analyzer.cli.result_status import renderStatuses

    analysis = loadAnalysis(configuration)
    dataset_repo, era_repo = getRepos(
        analysis.extra_dataset_paths, analysis.extra_era_paths
    )
    all_samples = getSamples(analysis, dataset_repo)

    ret = checkResults(files)
    renderStatuses(ret, all_samples, only_bad=only_bad)


@cli.command()
@click.argument("inputs", type=str, nargs=-1)
@click.option("--output", "-o", type=str, required=True)
@click.option("--configuration", "-c", type=str, required=True)
@click.option("--executor", "-e", type=str, required=True)
def patch(
    inputs,
    output,
    configuration,
    executor,
):
    from analyzer.core.running import patchFromPath

    patchFromPath(configuration, inputs, output, executor)


@cli.command()
@click.argument("inputs", type=str, nargs=-1)
@click.option("--interpretter", is_flag=True)
def browse(inputs, interpretter):
    from analyzer.core.results import loadResults
    from analyzer.core.serialization import setupConverter, converter

    setupConverter(converter)
    res = loadResults(inputs)
    if interpretter:
        jumpIn(results=res)
    else:
        from analyzer.cli.browser import ResultBrowser

        browser = ResultBrowser(res)
        browser.run()


@cli.group()
def cache():
    pass


@cache.command()
def clear():
    from analyzer.core.caching import cache

    cache.clear()


@cache.command()
def list():
    from analyzer.core.caching import cache

    for f in cache:
        print(f)


@cli.group("list")
def listData():
    pass


@click.option("--filter", type=str)
@click.option("--csv", is_flag=True)
@listData.command()
def samples(filter, csv):
    from analyzer.cli.dataset_table import createSampleTable, createDatasetTable
    from analyzer.utils.querying import Pattern
    from analyzer.core.running import getRepos

    if filter:
        filter_pattern = Pattern(filter)
    else:
        filter_pattern = None
    dataset_repo, era_repo = getRepos()
    table = createSampleTable(dataset_repo, pattern=filter_pattern, as_csv=csv)
    print(table)


@click.option("--filter", type=str)
@click.option("--csv", is_flag=True)
@listData.command()
def datasets(filter, csv):
    from analyzer.cli.dataset_table import createSampleTable, createDatasetTable
    from analyzer.utils.querying import Pattern
    from analyzer.core.running import getRepos

    if filter:
        filter_pattern = Pattern(filter)
    else:
        filter_pattern = None
    dataset_repo, era_repo = getRepos()
    table = createDatasetTable(dataset_repo, pattern=filter_pattern, as_csv=csv)
    print(table)


@listData.group()
def eras():
    pass


def main():
    from analyzer.logging import setupLogging

    setupLogging()
    cli()
