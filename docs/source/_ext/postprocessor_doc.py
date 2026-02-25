import inspect
import pkgutil
import importlib

from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.errors import ExtensionError

import sys
import os

_LOADED_RESULTS_CACHE = {}


class PostprocessorModulesDirective(SphinxDirective):
    def run(self):
        from analyzer.postprocessing.processors import BasePostprocessor
        from analyzer.postprocessing.transforms.registry import TransformHistogram
        import analyzer.postprocessing
        import attrs

        def is_target_class(cls, base_class):
            return (
                inspect.isclass(cls)
                and issubclass(cls, base_class)
                and cls is not base_class
                and not cls.__name__.startswith("Base")
            )

        def get_classes(base_class, module_path, prefix):
            modules = {}
            for loader, module_name, is_pkg in pkgutil.walk_packages(
                module_path, prefix
            ):
                if is_pkg:
                    continue
                try:
                    mod = importlib.import_module(module_name)
                    classes = []
                    for name, obj in inspect.getmembers(mod):
                        if (
                            is_target_class(obj, base_class)
                            and obj.__module__ == module_name
                        ):
                            classes.append(obj)

                    if classes:
                        parts = module_name.split(".")
                        group_name = parts[-1]
                        if group_name not in modules:
                            modules[group_name] = []
                        modules[group_name].extend(classes)
                except Exception as e:
                    print(f"Error loading {module_name}: {e}")
            return modules

        rst = []
        bases_to_document = [
            (
                "Postprocessors",
                BasePostprocessor,
                analyzer.postprocessing.__path__,
                analyzer.postprocessing.__name__ + ".",
            ),
            (
                "Histogram Transforms",
                TransformHistogram,
                [os.path.join(analyzer.postprocessing.__path__[0], "transforms")],
                analyzer.postprocessing.__name__ + ".transforms.",
            ),
        ]

        for section_title, base_class, module_path, prefix in bases_to_document:
            rst.append(section_title)
            rst.append("=" * len(section_title))
            rst.append("")

            modules = get_classes(base_class, module_path, prefix)

            for group, classes in sorted(modules.items()):
                group_title = f"{group.capitalize()}"
                rst.append(group_title)
                rst.append("-" * len(group_title))
                rst.append("")

                for cls in sorted(classes, key=lambda c: c.__name__):
                    cls_name = cls.__name__
                    rst.append(cls_name)
                    rst.append("^" * len(cls_name))
                    rst.append("")

                    rst.append(f".. py:class:: {cls.__module__}.{cls_name}")
                    rst.append("   :noindex:")
                    rst.append("")

                    doc = inspect.getdoc(cls)
                    if doc:
                        lines = doc.splitlines()
                        clean_doc = []
                        for line in lines:
                            if line.strip() == "Parameters":
                                break
                            clean_doc.append(line)

                        doc_text = "\n".join(clean_doc).strip()
                        if doc_text:
                            for line in doc_text.splitlines():
                                rst.append(f"   {line}")
                            rst.append("")

                    if attrs.has(cls):
                        fields = attrs.fields(cls)
                        table_lines = [
                            "   .. list-table:: Configuration Variables",
                            "      :widths: 20 20 60",
                            "      :header-rows: 1",
                            "",
                            "      * - Name",
                            "        - Type",
                            "        - Default",
                        ]

                        has_fields = False
                        for f in fields:
                            if f.name.startswith("_"):
                                continue
                            has_fields = True

                            tname = "Any"
                            if f.type is not None:
                                if hasattr(f.type, "__name__"):
                                    tname = f.type.__name__
                                else:
                                    tname = str(f.type)
                            tname = tname.replace("typing.", "").replace(
                                "analyzer.core.columns.", ""
                            )

                            t_str = f"``{tname}``".replace("\n", " ")

                            if f.default == attrs.NOTHING:
                                def_str = "**Required**"
                            elif isinstance(f.default, attrs.Factory):  # type: ignore
                                def_str = "*Factory*"
                            else:
                                raw_def = (
                                    repr(f.default).replace("\n", " ").replace("\r", "")
                                )
                                def_str = f"``{raw_def}``"

                            table_lines.append(f"      * - ``{f.name}``")
                            table_lines.append(f"        - {t_str}")
                            table_lines.append(f"        - {def_str}")

                        if has_fields:
                            table_lines.append("")
                            rst.extend(table_lines)

        vl = ViewList()
        for i, line in enumerate(rst):
            vl.append(line, "<postprocessor-modules>", i)

        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, vl, node)
        return node.children


class PostprocessorExampleDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0

    def run(self):
        yaml_path = self.arguments[0]

        env = self.state.document.settings.env
        rel_fn, abs_fn = env.relfn2path(yaml_path)
        env.note_dependency(rel_fn)

        with open(abs_fn, "r") as f:
            yaml_content = f.read()

        import os
        from analyzer.postprocessing.running import runPostprocessors

        example_name = os.path.splitext(os.path.basename(abs_fn))[0]
        static_dir = os.path.join(
            env.app.srcdir, "_static", "generated_examples", example_name
        )
        os.makedirs(static_dir, exist_ok=True)
        prefix = static_dir

        repo_root = os.path.abspath(os.path.join(env.app.srcdir, "../.."))
        example_data_path = os.path.join(
            repo_root, "analyzer_resources", "example_data", "example_result.result"
        )

        try:
            if example_data_path not in _LOADED_RESULTS_CACHE:
                from analyzer.core.results import loadResults

                print(f"Loading results into cache for {example_data_path}...")
                _LOADED_RESULTS_CACHE[example_data_path] = loadResults(
                    [example_data_path]
                )

            runPostprocessors(
                abs_fn,
                [example_data_path],
                prefix=prefix,
                parallel=1,
                loaded_results=_LOADED_RESULTS_CACHE[example_data_path],
            )
        except Exception as e:
            print(f"Failed to run postprocessor example {abs_fn}: {e}")

        code_block = nodes.literal_block(yaml_content, yaml_content)
        code_block["language"] = "yaml"

        details_start = nodes.raw(
            "",
            '<details><summary style="cursor: pointer; font-weight: bold; margin-bottom: 0.5em;">Show YAML Configuration</summary>\n',
            format="html",
        )
        details_end = nodes.raw("", "</details>\n", format="html")

        result_nodes = [details_start, code_block, details_end]

        def find_output_files(dir_path):
            images = []
            tex_files = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    if file.endswith(".png"):
                        images.append(full_path)
                    elif file.endswith(".tex"):
                        tex_files.append(full_path)
            return sorted(images), sorted(tex_files)

        images, tex_files = find_output_files(static_dir)

        if images:
            for img in images:
                img_rel = os.path.relpath(img, env.app.srcdir)
                img_uri = "/" + img_rel.replace(os.sep, "/")
                result_nodes.append(nodes.image(uri=img_uri))

        if tex_files:
            for tex_path in tex_files:
                with open(tex_path, "r") as f:
                    tex_code = f.read()

                details_start_tex = nodes.raw(
                    "",
                    '<details open><summary style="cursor: pointer; font-weight: bold; margin-bottom: 0.5em;">Generated LaTeX</summary>\n',
                    format="html",
                )
                tex_block = nodes.literal_block(tex_code, tex_code)
                tex_block["language"] = "latex"
                details_end_tex = nodes.raw("", "</details>\n", format="html")

                result_nodes.extend([details_start_tex, tex_block, details_end_tex])

        return result_nodes


def setup(app):
    app.add_directive("postprocessor-modules", PostprocessorModulesDirective)
    app.add_directive("postprocessor-example", PostprocessorExampleDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
