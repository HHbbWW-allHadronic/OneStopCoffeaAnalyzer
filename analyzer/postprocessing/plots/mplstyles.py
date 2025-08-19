from pathlib import Path
import matplotlib.font_manager as font_manager
import mplhep
from analyzer.configuration import CONFIG

mplhep.style.use("CMS")


def loadStyles():
    font_dirs = [str(Path(CONFIG.STATIC_PATH) / "fonts")]
    str(Path(CONFIG.STYLE_PATH) / "style.mplstyle")
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
