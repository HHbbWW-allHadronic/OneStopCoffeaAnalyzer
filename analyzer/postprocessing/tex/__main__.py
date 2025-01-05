import sys
from . import renderTemplate
from pathlib import Path


data = [{"path": x} for x in (Path("postprocessing").rglob("*.png"))]
renderTemplate(
    "analyzer_resources/templates/image_grid.tex",
    "testrender/test.tex",
    dict(images=data),
)
