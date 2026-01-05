from pathlib import Path
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import mplhep
from analyzer.configuration import CONFIG


def loadStyles():
    font_dir = str(Path(CONFIG.post.static_resource_path) / "fonts")
    mplhep.style.use("CMS")
    # str(Path(CONFIG.STYLE_PATH) / "style.mplstyle")
    mpl.rcParams["figure.constrained_layout.use"] = True
    font_files = font_manager.findSystemFonts(fontpaths=[font_dir])
    for font in font_files:
        font_manager.fontManager.addfont(font)
