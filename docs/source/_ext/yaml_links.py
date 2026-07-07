import re
from docutils import nodes


def buildModuleMap(app, env):
    module_map = {}
    py_objects = env.domaindata.get("py", {}).get("objects", {})
    for full_name, (docname, node_id, _, _) in py_objects.items():
        class_name = full_name.split(".")[-1]
        module_map[class_name] = (docname, node_id)

    env.yaml_module_map = module_map


def injectLinks(app, html_string, module_map, current_docname):
    pattern = re.compile(
        r'(module_name|name)(</span>\s*<span\s+class="p">:</span>\s*<span\s+class="[a-zA-Z0-9\- ]*">\s*</span>\s*<span\s+class="[a-zA-Z0-9\- ]*">)([A-Za-z0-9_]+)(</span>)'
    )

    def replacer(match):
        key = match.group(1)
        middle_spans = match.group(2)
        module_name = match.group(3)
        closing_span = match.group(4)

        if module_name in module_map:
            target_docname, node_id = module_map[module_name]
            uri = app.builder.get_relative_uri(current_docname, target_docname)
            link = f'<a href="{uri}#{node_id}" class="reference internal">{module_name}</a>'
            return f"{key}{middle_spans}{link}{closing_span}"

        return match.group(0)

    ret = pattern.sub(replacer, html_string)
    return ret


def processDoctree(app, doctree, docname):
    if app.builder.format != "html":
        return

    module_map = getattr(app.env, "yaml_module_map", {})

    for node in doctree.traverse(nodes.literal_block):
        if node.get("language") == "yaml":
            # Run Sphinx's configured syntax highlighter on the block
            html_string = app.builder.highlighter.highlight_block(
                node.rawsource, "yaml", location=node
            )

            # Inject our <a> tags
            new_html = injectLinks(app, html_string, module_map, docname)

            # Replace the generic literal_block with our custom raw HTML node
            raw_node = nodes.raw("", new_html, format="html")
            node.replace_self(raw_node)


def setup(app):
    app.connect("env-updated", buildModuleMap)
    app.connect("doctree-resolved", processDoctree)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
