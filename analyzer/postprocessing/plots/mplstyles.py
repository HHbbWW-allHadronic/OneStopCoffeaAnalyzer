from pathlib import Path
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import mplhep
from analyzer.configuration import CONFIG


def loadStyles():
    font_dirs = [str(Path(CONFIG.STATIC_PATH) / "fonts")]
    mplhep.style.use("CMS")
    # str(Path(CONFIG.STYLE_PATH) / "style.mplstyle")
    mpl.rcParams["figure.constrained_layout.use"] = True
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
