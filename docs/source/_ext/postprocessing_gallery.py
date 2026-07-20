import hashlib
import os
from pathlib import Path
import logging
import copy

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

_results_cache = {}


class PostprocessGalleryNode(nodes.General, nodes.Element):
    pass


def getProjectRoot(env):
    root = env.config.postprocess_gallery_root
    if root is None:
        root = str(Path(env.srcdir).parent.parent)
    return Path(root)


def getResults(results_dir):
    key = str(Path(results_dir).resolve())
    if key not in _results_cache:
        from analyzer.core.results import loadResults

        files = sorted(str(f) for f in Path(results_dir).glob("*.result"))
        if not files:
            raise FileNotFoundError(f"No .result files in {results_dir}")
        _results_cache[key] = loadResults(files)
    return copy.deepcopy(_results_cache[key])


def runPostprocessor(yaml_text, results_dir, output_dir):
    import matplotlib as mpl

    mpl.use("Agg")

    from analyzer.postprocessing.running import loadPostprocessor, runResults
    from analyzer.postprocessing.style import loadStyles
    import analyzer.postprocessing.plots.utils as plot_utils

    loadStyles()
    plot_utils.INCLUDE_SIDECAR = False

    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "_config.yaml"
    config_file.write_text(yaml_text)

    postprocessor = loadPostprocessor(str(config_file))
    results = getResults(results_dir)
    logger.info(f"Generating results in {output_dir}")
    runResults(postprocessor, results, prefix=str(output_dir))
    logger.info(f"Finished generating results")


class PostprocessDirective(SphinxDirective):
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = True
    option_spec = {
        "results": directives.unchanged_required,
        "columns": directives.nonnegative_int,
        "caption": directives.unchanged,
        "category": directives.unchanged,
        "title": directives.unchanged,
        "hide-yaml": directives.flag,
    }

    def run(self):
        root = getProjectRoot(self.env)

        if self.arguments:
            config_path = root / self.arguments[0]
            yaml_text = config_path.read_text()
        else:
            yaml_text = "\n".join(self.content)

        results_dir = root / self.options["results"]
        yaml_hash = hashlib.md5(yaml_text.encode()).hexdigest()[:12]
        output_dir = Path(self.env.srcdir) / "_static" / "gallery_generated" / yaml_hash

        if not output_dir.exists() or not list(output_dir.glob("*.png")):
            try:
                runPostprocessor(yaml_text, results_dir, output_dir)
            except Exception as e:
                logger.warning("postprocess directive failed: %s", e)
                error = self.state.document.reporter.warning(
                    f"Postprocessor execution failed: {e}",
                    line=self.lineno,
                )
                raise
                return [error]

        images = sorted(output_dir.glob("*.png"))

        if not hasattr(self.env, "postprocess_entries"):
            self.env.postprocess_entries = []

        title = self.options.get("title")
        anchor = nodes.make_id(title) if title else ""

        if title and images:
            self.env.postprocess_entries.append(
                {
                    "title": title,
                    "category": self.options.get("category", "General"),
                    "thumbnail": str(images[0]),
                    "docname": self.env.docname,
                    "anchor": anchor,
                }
            )

        wrapper = nodes.container(classes=["postprocess-example"])
        if anchor:
            wrapper["ids"] = [anchor]

        if "hide-yaml" not in self.options:
            code_node = nodes.literal_block(yaml_text, yaml_text)
            code_node["language"] = "yaml"
            wrapper += code_node

        columns = self.options.get("columns", 2)
        grid = nodes.container(
            classes=["postprocess-image-grid", f"postprocess-cols-{columns}"]
        )

        if self.options.get("caption"):
            caption_para = nodes.paragraph(
                text=self.options["caption"], classes=["postprocess-grid-caption"]
            )
            wrapper += caption_para

        doc_dir = (Path(self.env.srcdir) / self.env.docname).parent
        for img_path in images:
            rel_path = os.path.relpath(str(img_path), str(doc_dir))
            fig = nodes.figure("", classes=["postprocess-plot"])
            img = nodes.image("", uri=rel_path)
            img.candidates = {"?": None}
            img["candidates"] = {}
            img["candidates"]["*"] = rel_path
            img["alt"] = img_path.stem.replace("_", " ")
            fig += img
            fig += nodes.caption("", nodes.Text(img_path.stem))
            grid += fig

        wrapper += grid

        return [wrapper]


class PostprocessGalleryDirective(SphinxDirective):
    has_content = False
    option_spec = {}

    def run(self):
        node = PostprocessGalleryNode()
        node.document = self.state.document
        return [node]


def resolveGalleryNodes(app, doctree, docname):
    if app.builder.format != "html":
        return

    for node in doctree.traverse(PostprocessGalleryNode):
        entries = getattr(app.env, "postprocess_entries", [])
        if not entries:
            node.replace_self([])
            continue

        categories = {}
        for entry in entries:
            cat = entry.get("category", "General")
            categories.setdefault(cat, []).append(entry)

        content = []
        for cat_name, cat_entries in sorted(categories.items()):
            content.append(
                nodes.rubric("", cat_name, classes=["postprocess-category-heading"])
            )

            grid = nodes.container(classes=["postprocess-gallery-grid"])
            for entry in cat_entries:
                card = nodes.container(classes=["postprocess-card"])

                if entry.get("thumbnail"):
                    doc_dir = str((Path(app.srcdir) / docname).parent)
                    thumb_rel = os.path.relpath(entry["thumbnail"], doc_dir)
                    img = nodes.image(
                        "", uri=thumb_rel, classes=["postprocess-card-image"]
                    )
                    img["candidates"] = {}
                    img["candidates"]["*"] = thumb_rel
                    card += img

                ref_uri = app.builder.get_relative_uri(docname, entry["docname"])
                anchor = entry.get("anchor", "")
                full_uri = f"{ref_uri}#{anchor}" if anchor else ref_uri

                ref = nodes.reference(
                    "",
                    entry.get("title", ""),
                    refuri=full_uri,
                    classes=["postprocess-card-link"],
                )
                card += nodes.paragraph(
                    "", "", ref, classes=["postprocess-card-title-text"]
                )
                grid += card

            content.append(grid)

        node.replace_self(content)


def purgeEntries(app, env, docname):
    if hasattr(env, "postprocess_entries"):
        env.postprocess_entries = [
            e for e in env.postprocess_entries if e["docname"] != docname
        ]


def mergeEntries(app, env, docnames, other):
    if not hasattr(env, "postprocess_entries"):
        env.postprocess_entries = []
    env.postprocess_entries.extend(getattr(other, "postprocess_entries", []))


def setup(app):
    app.add_config_value("postprocess_gallery_root", None, "env")
    app.add_node(PostprocessGalleryNode)
    app.add_directive("postprocess", PostprocessDirective)
    app.add_directive("postprocess-gallery", PostprocessGalleryDirective)
    app.connect("doctree-resolved", resolveGalleryNodes)
    app.connect("env-purge-doc", purgeEntries)
    app.connect("env-merge-info", mergeEntries)
    app.add_css_file("postprocess_gallery.css")

    return {
        "version": "0.1",
        "parallel_read_safe": False,
        "parallel_write_safe": True,
    }
