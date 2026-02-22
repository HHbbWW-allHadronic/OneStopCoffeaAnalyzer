import inspect
import pkgutil
import importlib

from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


class AnalyzerModulesDirective(SphinxDirective):
    def run(self):
        import analyzer.modules
        from analyzer.core.analysis_modules import BaseAnalyzerModule
        import attrs

        def is_analyzer_module(cls):
            return (
                inspect.isclass(cls)
                and issubclass(cls, BaseAnalyzerModule)
                and cls is not BaseAnalyzerModule
                and not cls.__name__.startswith("Base")
            )

        modules = {}
        prefix = analyzer.modules.__name__ + "."
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            analyzer.modules.__path__, prefix
        ):
            if is_pkg:
                continue
            try:
                mod = importlib.import_module(module_name)
                classes = []
                for name, obj in inspect.getmembers(mod):
                    if is_analyzer_module(obj) and obj.__module__ == module_name:
                        classes.append(obj)

                if classes:
                    parts = module_name.split(".")
                    group_name = parts[2]
                    file_name = parts[-1]
                    if group_name not in modules:
                        modules[group_name] = {}
                    modules[group_name][file_name] = classes
            except Exception as e:
                print(f"Error loading {module_name}: {e}")

        rst = []
        for group, files in sorted(modules.items()):
            group_title = f"{group.capitalize()} Modules"
            rst.append(group_title)
            rst.append("-" * len(group_title))
            rst.append("")

            for file_name, classes in sorted(files.items()):
                file_title = f"{file_name}.py"
                rst.append(file_title)
                rst.append("^" * len(file_title))
                rst.append("")

                for cls in sorted(classes, key=lambda c: c.__name__):
                    # We output a standard section for each class instead of py:class
                    # to fully avoid the ugly signature rendering
                    cls_name = cls.__name__
                    rst.append(cls_name)
                    rst.append('"' * len(cls_name))
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
            vl.append(line, "<analyzer-modules>", i)

        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, vl, node)
        return node.children


def setup(app):
    app.add_directive("analyzer-modules", AnalyzerModulesDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
