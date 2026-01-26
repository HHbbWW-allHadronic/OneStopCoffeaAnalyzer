import yaml
import urllib.parse
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


DAS_BASE_URL = "https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input={}"
XSDB_BASE_URL = (
    "https://xsecdb-xsdb-official.app.cern.ch/xsdb/?searchQuery=process_name%3D{}"
)


def loadDatasets(root_path: Path) -> list[dict[str, Any]]:
    datasets = []
    if not root_path.exists():
        return datasets

    for yaml_file in root_path.rglob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                content = yaml.safe_load(f)
                if isinstance(content, list):
                    datasets.extend(content)
        except Exception as e:
            print(f"Error reading {yaml_file}: {e}")
    return datasets


def groupDatasets(
    datasets: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {
        "Data": {},
        "Signal": {},
        "Background": {},
    }

    for ds in datasets:
        name = ds.get("name", "")
        sample_type = ds.get("sample_type", "Unknown")
        era = str(ds.get("era", "Unknown"))

        category = "Background"
        if sample_type == "Data":
            category = "Data"
        elif "signal" in name.lower() or sample_type == "Signal":
            category = "Signal"

        process_name = name
        if era in process_name and len(process_name) > len(era):
            process_name = (
                process_name.replace(f"_{era}", "").replace(era, "").strip("_")
            )

        if process_name not in grouped[category]:
            grouped[category][process_name] = {}

        grouped[category][process_name][era] = ds

    return grouped


def generateDasLinkNode(das_path: str) -> nodes.reference:
    url = DAS_BASE_URL.format(urllib.parse.quote(das_path))
    return nodes.reference(refuri=url, text="[DAS]")


def generateXsdbLinkNode(das_path: str) -> nodes.reference:
    parts = das_path.strip("/").split("/")
    if not parts:
        return None

    process_name = parts[0]
    url = XSDB_BASE_URL.format(urllib.parse.quote(process_name))
    return nodes.reference(refuri=url, text="[XSDB]")


class DatasetListDirective(SphinxDirective):
    def run(self) -> list[nodes.Node]:
        dataset_root = Path(__file__).parents[3] / "analyzer_resources" / "datasets"

        if not dataset_root.exists():
            return [
                nodes.error(
                    None,
                    nodes.paragraph(
                        text=f"Dataset directory not found: {dataset_root}"
                    ),
                )
            ]

        datasets = loadDatasets(dataset_root)
        grouped_datasets = groupDatasets(datasets)

        return self.renderContent(grouped_datasets)

    def renderContent(
        self, grouped_datasets: dict[str, dict[str, dict[str, Any]]]
    ) -> list[nodes.Node]:
        content_nodes = []

        for category, processes in grouped_datasets.items():
            if not processes:
                continue

            cat_section = nodes.section(ids=[category.lower().replace(" ", "-")])
            cat_section += nodes.title(text=category)

            for process, periods in sorted(processes.items()):
                proc_id = (
                    f"{category}-{process}".lower().replace(" ", "-").replace("_", "-")
                )
                proc_section = nodes.section(ids=[proc_id])
                proc_section += nodes.title(text=process)

                for era, ds_data in sorted(periods.items()):
                    proc_section += self.renderDatasetEntry(era, ds_data)

                cat_section += proc_section

            content_nodes.append(cat_section)

        return content_nodes

    def renderDatasetEntry(self, era: str, ds_data: dict[str, Any]) -> nodes.Node:
        container = nodes.container()
        container += nodes.rubric(text=f"Era: {era}")

        info_list = nodes.bullet_list()

        title = ds_data.get("title", "")
        if title:
            item = nodes.list_item()
            item += nodes.paragraph(text=f"Title: {title}")
            info_list += item

        samples = ds_data.get("samples", [])
        if samples:
            samples_item = nodes.list_item()
            samples_item += nodes.paragraph(text="Samples:")
            samples_list = nodes.bullet_list()

            for sample in samples:
                samples_list += self.renderSampleItem(sample)

            samples_item += samples_list
            info_list += samples_item

        container += info_list
        return container

    def renderSampleItem(self, sample: dict[str, Any]) -> nodes.list_item:
        sample_name = sample.get("name", "Unknown Sample")
        das_path = sample.get("das_path", "")

        item = nodes.list_item()
        para = nodes.paragraph()
        para += nodes.Text(f"{sample_name} ")

        if das_path:
            para += generateDasLinkNode(das_path)
            para += nodes.Text(" ")

            xsdb_node = generateXsdbLinkNode(das_path)
            if xsdb_node:
                para += xsdb_node

        item += para
        return item


def setup(app):
    app.add_directive("dataset-list", DatasetListDirective)
    return {
        "version": "0.5",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
