def renderStatuses(statuses, all_samples, threshold=1.0, only_bad=False):
    from rich.console import Console
    from rich.style import Style
    from rich.table import Table

    console = Console()
    present = set((x.dataset_name, x.sample_name) for x in statuses)
    missing = sorted(all_samples - present)

    table = Table(title="Processing Status")
    for x in ("Dataset Name", "Sample Name", "% Complete", "Processed", "Total"):
        table.add_column(x)

    for status in statuses:
        expected = status.events_expected
        found = status.events_found
        expected - found
        percent = round(found / expected * 100, 2)
        frac_done = found / expected
        done = (frac_done >= threshold) and (frac_done <= 1.0)

        if only_bad and done:
            continue

        table.add_row(
            status.dataset_name,
            status.sample_name,
            str(percent),
            f"{found}",
            f"{expected}",
            style=Style(color="green" if done else "red"),
        )
    for dataset_name, sample_name in missing:
        table.add_row(
            dataset_name,
            sample_name,
            "Missing",
            "Missing",
            "Missing",
            style=Style(color="red"),
        )

    console.print(table)
