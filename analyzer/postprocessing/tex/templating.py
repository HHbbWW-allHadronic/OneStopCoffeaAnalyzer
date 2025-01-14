from pathlib import Path
import jinja2
from analyzer.configuration import CONFIG

latex_jinja_env = jinja2.Environment(
    block_start_string="\BLOCK{",
    block_end_string="}",
    variable_start_string="\VAR{",
    variable_end_string="}",
    comment_start_string="\#{",
    comment_end_string="}",
    line_statement_prefix="%-",
    line_comment_prefix="%#",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(CONFIG.TEMPLATE_PATH),
)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def renderTemplate(template_path, outpath, data):
    template = latex_jinja_env.get_template(template_path)
    rendered = template.render(**data)
    outpath = Path(outpath)
    outpath.parent.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        f.write(rendered)
