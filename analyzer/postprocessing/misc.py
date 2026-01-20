import csv
from pathlib import Path


def dumpYield(sectors, output):
    output = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Dataset",
                "raw_initial",
                "raw_final",
                "raw_eff",
                "weighted_initial",
                "weighted_final",
                "weighted_eff",
            ]
        )
        for sector in sectors:
            selection = sector.result.selection_flow
            raw_selection = sector.result.raw_selection_flow
            title = sector.sector_params.dataset.name
            initial = selection.total_events
            final = selection.final_events
            raw_initial = raw_selection.total_events
            raw_final = raw_selection.final_events
            eff = selection.selection_efficiency
            raw_eff = raw_selection.selection_efficiency

            writer.writerow(
                [title, raw_initial, raw_final, raw_eff, initial, final, eff]
            )
