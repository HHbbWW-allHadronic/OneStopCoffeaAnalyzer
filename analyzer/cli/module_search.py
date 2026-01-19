import difflib
import inspect
from rich.table import Table
from rich.console import Console
from analyzer.core.analysis_modules import AnalyzerModule
import analyzer.modules  # noqa


def searchModules(query: str):
    console = Console()

    def getSubclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in getSubclasses(c)]
        )

    modules = getSubclasses(AnalyzerModule)

    results = []

    query = query.lower()

    for mod in modules:
        name = mod.name()
        doc = inspect.getdoc(mod) or ""

        score = 0
        matches = []
        if query == name.lower():
            score += 100
            matches.append("Exact Name")
        elif query in name.lower():
            score += 50
            matches.append("Partial Name")
        name_ratio = difflib.SequenceMatcher(None, query, name.lower()).ratio()
        if name_ratio > 0.6:
            score += int(name_ratio * 40)
            matches.append(f"Fuzzy Name ({int(name_ratio * 100)}%)")
        if query in doc.lower():
            score += 30
            matches.append("Docstring")

        if score > 0:
            results.append(
                {
                    "name": name,
                    "score": score,
                    "matches": ", ".join(matches),
                    "doc": doc.split("\n")[0] if doc else "",
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    if not results:
        console.print(f"[yellow]No modules found matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Module Name", style="cyan", no_wrap=True)
    table.add_column("Match Type", style="magenta")
    table.add_column("Description", style="green")

    for res in results:
        table.add_row(res["name"], res["matches"], res["doc"])

    console.print(table)
