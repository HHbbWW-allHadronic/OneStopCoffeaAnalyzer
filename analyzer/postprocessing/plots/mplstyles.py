from pathlib import Path
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from analyzer.configuration import CONFIG

def loadStyles():
    font_dirs = [str(Path(CONFIG.STATIC_PATH)/"fonts")]
    style = str(Path(CONFIG.STYLE_PATH) / "style.mplstyle")
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
    plt.style.use(style)

