# Builds Construct's Qt theme (dist/qt_themes/construct) from qdarkstyle, styled after the look of
# Hyrule Warriors: Age of Imprisonment's menus (original work, no game assets): near-black stone
# panels with a teal tint, thin gold frames, parchment text, green-teal "construct energy" for focus
# and progress, and serif headings. No picture strips on the bars.
import pathlib
import re
import shutil
import sys

from PIL import Image

root = pathlib.Path(sys.argv[1])  # construct source root
themes = root / 'dist' / 'qt_themes'
src = themes / 'qdarkstyle'
out = themes / 'construct'
(out / 'rc').mkdir(parents=True, exist_ok=True)
(out / 'icons' / '16x16').mkdir(parents=True, exist_ok=True)
for n in ('helix_top.png', 'helix_top@2x.png', 'helix_bottom.png', 'helix_bottom@2x.png', 'bg.png'):
    if (out / n).exists():
        (out / n).unlink()
# original decorative art from make_construct_art.py (argv[2])
art_dir = pathlib.Path(sys.argv[2])
ART = []  # stored under rc/ as art_*.png (images under the construct_qss prefix load reliably)
for n in ('glyphs', 'frame'):
    for suffix in ('', '@2x'):
        if not (art_dir / f'{n}{suffix}.png').exists():
            continue
        if (out / f'{n}{suffix}.png').exists():
            (out / f'{n}{suffix}.png').unlink()
        shutil.copyfile(art_dir / f'{n}{suffix}.png', out / 'rc' / f'art_{n}{suffix}.png')
        ART.append(f'rc/art_{n}{suffix}.png')
for stale in (out / 'rc').glob('art_*.png'):
    if f'rc/{stale.name}' not in ART:
        stale.unlink()

# palette
BG, BG_DEEP, BG_ALT, BG_DARKEST = '#111a1c', '#0b1214', '#142022', '#080d0e'
STONE, STONE_DIM, STONE_DARK = '#3d5753', '#2a3f3d', '#203130'
TEXT, TEXT_BRIGHT = '#ece2c6', '#fff6dc'
GOLD, GOLD_BRIGHT, GOLD_DIM, GOLD_DEEP = '#c9a14a', '#e8c86c', '#8a6a2a', '#4a3a18'
GLOW = '#46e3bf'
TEAL, TEAL_DIM, JADE, BRONZE, EMBER = '#46e3bf', '#2a8a76', '#2fbf8f', '#b0703a', '#d66040'
GREY, GREY_DIM = '#9aa59f', '#62706b'

COLORS = {
    # backgrounds
    '#31363b': BG, '#232629': BG_DEEP, '#2a2929': BG_ALT, '#2c2f32': BG_ALT,
    '#2b2e31': BG_ALT, '#2a2f33': BG_ALT, '#1f2225': BG_DARKEST,
    # frames and borders: teal stone
    '#76797c': STONE, '#54575b': STONE_DIM, '#565a5e': '#34504c', '#4a4a4a': '#2e4644',
    '#454545': '#2c4341', '#434242': '#2a403e', '#403f3f': '#283d3b', '#3a3939': STONE_DARK,
    '#393e43': '#233634', '#393838': STONE_DARK, '#4f515b': '#314a47', '#58595a': '#34504c',
    # text: parchment
    '#eff0f1': TEXT, '#f0f0f0': TEXT, '#ffffff': TEXT_BRIGHT, '#fff': TEXT_BRIGHT,
    # blue accent -> gold
    '#3daee9': GOLD, '#18465d': GOLD_DEEP, '#3375a3': GOLD_DIM, '#287399': '#7a5e26',
    '#334e5e': '#3e3216', '#05b8cc': GOLD_BRIGHT, '#00ccdd': GOLD_BRIGHT,
    # greys -> stone greys
    '#ccc': '#cfd3c8', '#656565': GREY_DIM, '#666': GREY_DIM, '#6a6969': '#66756f',
    '#605f5f': '#5c6a65', '#929192': GREY, '#787876': '#7a8883', '#5a7566': '#1d2e2d',
}
RGBA = {
    (41, 41, 41): (13, 21, 22), (44, 44, 44): (16, 25, 27), (71, 69, 71): (44, 66, 63),
    (80, 79, 80): (50, 74, 71), (95, 94, 95): (64, 90, 86), (58, 61, 66): (35, 54, 52),
    (240, 240, 240): (236, 226, 198), (255, 255, 255): (255, 246, 220), (56, 189, 225): (201, 161, 74),
    (1, 253, 201): (70, 227, 191), (150, 150, 150): (154, 165, 159), (144, 144, 144): (146, 158, 152),
}

qss = (src / 'style.qss').read_text(encoding='utf-8')
qss = re.sub(r'#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b', lambda m: COLORS.get(m.group(0).lower(), m.group(0)), qss)


def rgba_sub(m):
    new = RGBA.get((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return f'rgba({new[0]}, {new[1]}, {new[2]}, {m.group(4)})' if new else m.group(0)


qss = re.sub(r'rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\s*\)', rgba_sub, qss)
qss = qss.replace('url(:/qss_icons/rc/', 'url(:/construct_qss/rc/')
qss = qss.replace('color: white;', f'color: {TEXT_BRIGHT};')

SERIF = '"Palatino Linotype", "Book Antiqua", "Palatino", "Georgia", "URW Palladio L", serif'
extra = f'''
/* ---- Construct: Age of Imprisonment style ---- */
QMainWindow, QDialog {{
    background-color: {BG_DEEP};
}}
QMenuBar {{
    font-family: {SERIF};
    font-size: 10.5pt;
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #18262a, stop:1 {BG_DARKEST});
    border-bottom: 1px solid {GOLD_DIM};
    padding: 1px 4px;
}}
QMenuBar::item {{
    padding: 5px 12px;
    background: transparent;
    color: {TEXT};
}}
QMenuBar::item:selected {{
    color: {GOLD_BRIGHT};
    border-bottom: 2px solid {GOLD};
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(201, 161, 74, 0), stop:1 rgba(201, 161, 74, 60));
}}
QMenuBar::item:pressed {{
    color: {GOLD_BRIGHT};
    background-color: {GOLD_DEEP};
}}
QMenu {{
    background-color: {BG_DEEP};
    border: 1px solid {GOLD_DIM};
    padding: 4px 0;
}}
QMenu::item {{
    padding: 5px 28px 5px 22px;
    border-left: 2px solid transparent;
}}
QMenu::item:selected {{
    color: {GOLD_BRIGHT};
    border-left: 2px solid {GOLD};
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(201, 161, 74, 70), stop:1 rgba(201, 161, 74, 0));
}}
QMenu::separator {{
    height: 1px;
    margin: 4px 10px;
    background-color: {GOLD_DEEP};
}}
QStatusBar {{
    min-height: 26px;
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {BG_DARKEST}, stop:1 #18262a);
    border-top: 1px solid {GOLD_DIM};
    color: {TEXT};
}}
QStatusBar QLabel, QStatusBar QPushButton, QStatusBar QToolButton, QStatusBar QFrame {{
    background-color: transparent;
}}
QStatusBar::item {{
    border: none;
    border-left: 1px solid {STONE_DIM};
}}

/* game list */
QTreeView, QListView, QTableView {{
    background-color: {BG_DEEP};
    alternate-background-color: {BG};
    border: 1px solid {STONE_DIM};
    selection-background-color: {GOLD_DEEP};
}}
QTreeView::item, QListView::item {{
    padding: 2px 0;
    border-top: 1px solid transparent;
    border-bottom: 1px solid transparent;
}}
QTreeView::item:hover, QListView::item:hover, QTableView::item:hover {{
    background-color: rgba(70, 227, 191, 22);
    border-top: 1px solid rgba(70, 227, 191, 60);
    border-bottom: 1px solid rgba(70, 227, 191, 60);
    color: {TEXT_BRIGHT};
}}
QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{
    color: {GOLD_BRIGHT};
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(201, 161, 74, 95), stop:0.6 rgba(201, 161, 74, 35), stop:1 rgba(201, 161, 74, 10));
    border-top: 1px solid {GOLD_DIM};
    border-bottom: 1px solid {GOLD_DIM};
}}
QHeaderView::section {{
    font-family: {SERIF};
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a292c, stop:1 {BG_DARKEST});
    color: {GOLD_BRIGHT};
    border: none;
    border-right: 1px solid {STONE_DIM};
    border-bottom: 1px solid {GOLD_DIM};
    padding: 4px 8px;
}}
QHeaderView::section:hover {{
    color: {TEXT_BRIGHT};
}}

/* buttons */
QPushButton {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1d2d30, stop:1 {BG_DARKEST});
    border: 1px solid {GOLD_DIM};
    border-radius: 2px;
    padding: 4px 14px;
    color: {TEXT};
}}
QPushButton:hover {{
    border: 1px solid {GOLD_BRIGHT};
    color: {GOLD_BRIGHT};
}}
QPushButton:focus {{
    border: 1px solid {GLOW};
}}
QPushButton:pressed {{
    background-color: {GOLD_DEEP};
    color: {TEXT_BRIGHT};
}}
QPushButton:disabled {{
    border: 1px solid {STONE_DARK};
    color: {GREY_DIM};
}}
QToolButton:hover {{
    border: 1px solid {GOLD_DIM};
}}

/* inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
    background-color: {BG_DARKEST};
    border: 1px solid {STONE};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {GLOW};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{
    border: 1px solid {GOLD_DIM};
}}

/* frames */
QGroupBox {{
    border: 1px solid {STONE_DIM};
    border-top: 1px solid {GOLD_DIM};
    margin-top: 14px;
    padding-top: 6px;
}}
QGroupBox::title {{
    font-family: {SERIF};
    color: {GOLD_BRIGHT};
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
}}
QTabWidget::pane {{
    border: 1px solid {STONE_DIM};
    border-top: 1px solid {GOLD_DIM};
}}
QTabBar::tab {{
    font-family: {SERIF};
}}
QTabBar::tab:selected {{
    color: {GOLD_BRIGHT};
    border-bottom: 2px solid {GOLD};
}}
QTabBar::tab:!selected:hover {{
    color: {TEXT_BRIGHT};
    border-bottom: 2px solid rgba(70, 227, 191, 140);
}}
QListWidget#selectorList::item:selected {{
    color: {GOLD_BRIGHT};
    border-left: 2px solid {GOLD};
}}

/* sliders, scrollbars, progress: construct energy */
QSlider::groove:horizontal {{
    height: 4px;
    background-color: {STONE_DARK};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1f8f78, stop:1 {GLOW});
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 12px;
    margin: -5px 0;
    background-color: {GOLD};
    border: 1px solid {GOLD_BRIGHT};
    border-radius: 6px;
}}
QScrollBar:vertical {{
    width: 12px;
    background-color: {BG_DARKEST};
    border: none;
}}
QScrollBar:horizontal {{
    height: 12px;
    background-color: {BG_DARKEST};
    border: none;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {STONE};
    border-radius: 4px;
    margin: 2px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background-color: {GOLD_DIM};
}}
QProgressBar {{
    background-color: {BG_DARKEST};
    border: 1px solid {GOLD_DIM};
    border-radius: 2px;
    text-align: center;
    color: {TEXT_BRIGHT};
}}
QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #17705e, stop:1 {GLOW});
}}
QToolTip {{
    background-color: {BG_DARKEST};
    border: 1px solid {TEAL_DIM};
    color: {TEXT_BRIGHT};
}}

/* ---- ornaments: gold frame corners (glyph ribbons are drawn by glyph_ribbon.cpp), more colours ---- */
QTreeView#GameList, QTreeView {{
    background-color: {BG_DEEP};
    alternate-background-color: rgba(70, 227, 191, 7);
}}
QGroupBox {{
    border: 14px solid transparent;
    border-radius: 0px;
    border-image: url(:/construct_qss/rc/art_frame.png) 16 16 16 16 stretch stretch;
    margin-top: 16px;
    padding: 2px;
}}
QGroupBox::title {{
    color: {GOLD_BRIGHT};
    left: 14px;
}}
QMenu {{
    border: 10px solid transparent;
    border-radius: 0px;
    border-image: url(:/construct_qss/rc/art_frame.png) 16 16 16 16 stretch stretch;
    background-color: {BG_DEEP};
}}
QPushButton {{
    border-bottom: 2px solid {BRONZE};
}}
QPushButton:hover {{
    border-bottom: 2px solid {GOLD_BRIGHT};
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #24383a, stop:1 #10191a);
}}
QTabBar::tab:selected {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(201, 161, 74, 55), stop:1 rgba(201, 161, 74, 0));
}}
QHeaderView::section {{
    border-bottom: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {BRONZE}, stop:0.5 {GOLD_BRIGHT}, stop:1 {BRONZE});
}}
QCheckBox:hover, QRadioButton:hover {{
    color: {TEAL};
}}
QSlider::sub-page:horizontal {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {JADE}, stop:0.7 {TEAL}, stop:1 {GOLD_BRIGHT});
}}
QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #17705e, stop:0.6 {TEAL}, stop:1 {GOLD_BRIGHT});
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background-color: {BRONZE};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    background-color: #0b1a19;
}}
QComboBox QAbstractItemView {{
    border: 1px solid {GOLD_DIM};
    selection-background-color: {GOLD_DEEP};
    selection-color: {GOLD_BRIGHT};
}}
QLabel[class="error"], QLabel#error {{
    color: {EMBER};
}}
'''
qss = ('/* Construct theme: Age of Imprisonment style - dark stone, gold frames, construct-energy glow '
       '(generated from qdarkstyle by make_construct_theme_hwai.py; original work, no game assets) */\n') + qss + extra
(out / 'style.qss').write_text(qss, encoding='utf-8')

# control images: gold checks, teal-glow focus, parchment arrows, stone frames
CHECK, FOCUS = (201, 161, 74), (70, 227, 191)
ARROW, FRAME, DIM = (236, 226, 198), (154, 165, 159), (78, 92, 88)


def target_for(name):
    if name.startswith('close') or name == 'transparent.png':
        return None
    if '_focus' in name:
        return FOCUS
    if 'disabled' in name:
        return DIM
    if name.startswith(('checkbox_checked', 'checkbox_indeterminate', 'radio_checked')):
        return CHECK
    if name.endswith('-on.png') or name.split('.')[0] in ('up_arrow', 'down_arrow', 'left_arrow', 'right_arrow'):
        return ARROW
    return FRAME


rc_files = []
for png in sorted((src / 'rc').glob('*.png')):
    target = target_for(png.name)
    dst = out / 'rc' / png.name
    if target is None:
        shutil.copyfile(png, dst)
    else:
        im = Image.open(png).convert('RGBA')
        px = im.load()
        lums = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
                for p in (px[x, y] for x in range(im.width) for y in range(im.height)) if p[3] > 0]
        top = max(lums) if lums else 1.0
        for y in range(im.height):
            for x in range(im.width):
                r, g, b, a = px[x, y]
                if a == 0:
                    continue
                k = 0.35 + 0.65 * ((0.299 * r + 0.587 * g + 0.114 * b) / top if top else 1.0)
                px[x, y] = (int(target[0] * k), int(target[1] * k), int(target[2] * k), a)
        im.save(dst)
    rc_files.append(png.name)

for icon in ('lock.png', 'view-refresh.png'):
    shutil.copyfile(themes / 'colorful_dark' / 'icons' / '16x16' / icon, out / 'icons' / '16x16' / icon)
(out / 'icons' / 'index.theme').write_text(
    '[Icon Theme]\nName=construct\nComment=Construct theme\nInherits=colorful\n'
    'Directories=16x16\n\n[16x16]\nSize=16\n', encoding='utf-8')

qrc = ['<!--', 'SPDX-FileCopyrightText: 2026 birdman311', 'SPDX-License-Identifier: MIT', '-->', '',
       '<RCC>', '    <qresource prefix="icons/construct">',
       '        <file alias="index.theme">icons/index.theme</file>',
       '        <file alias="16x16/lock.png">icons/16x16/lock.png</file>',
       '        <file alias="16x16/view-refresh.png">icons/16x16/view-refresh.png</file>',
       '    </qresource>', '', '    <qresource prefix="construct_qss">']
qrc += [f'        <file>rc/{n}</file>' for n in rc_files]
qrc += [f'        <file>{n}</file>' for n in ART]
qrc += ['    </qresource>', '    <qresource prefix="construct">', '        <file>style.qss</file>',
        '    </qresource>', '</RCC>', '']
(out / 'style.qrc').write_text('\n'.join(qrc), encoding='utf-8')
print('Age of Imprisonment style theme written to', out, '| images', len(rc_files))
