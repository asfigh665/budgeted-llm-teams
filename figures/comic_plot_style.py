"""Typography shared by data-derived figures; no data transformations."""
from pathlib import Path
from matplotlib import font_manager

FONTS = Path(__file__).resolve().parent / 'fonts'
for name in ['ComicNeue-Regular.ttf', 'ComicNeue-Bold.ttf',
             'SourceSans3-Regular.ttf', 'SourceSans3-Semibold.ttf']:
    font_manager.fontManager.addfont(str(FONTS / name))
HEADING = font_manager.FontProperties(fname=str(FONTS / 'ComicNeue-Bold.ttf'), size=10)

