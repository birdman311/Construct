# SPDX-FileCopyrightText: 2026 birdman311
# SPDX-License-Identifier: MIT
# Renders Construct's theme stylesheet on sample widgets (offscreen) and prints any Qt parse warnings.
import pathlib
import sys

from PySide6.QtCore import Qt, qInstallMessageHandler
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                               QMenuBar, QProgressBar, QPushButton, QSlider, QStatusBar, QTabWidget, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget, QLineEdit)

theme = pathlib.Path(sys.argv[1])
out = sys.argv[2]
msgs = []
qInstallMessageHandler(lambda mode, ctx, msg: msgs.append(msg))
app = QApplication(sys.argv[:1] + ['-platform', 'offscreen'] + (['-style', sys.argv[3]] if len(sys.argv) > 3 else []))
print('style', app.style().name())
qss = (theme / 'style.qss').read_text(encoding='utf-8')
base = theme.as_posix()
qss = qss.replace('url(:/construct/', f'url({base}/').replace('url(:/construct_qss/', f'url({base}/')
app.setStyleSheet(qss)

win = QMainWindow()
win.resize(1200, 760)
mb = QMenuBar()
for n in ('File', 'Emulation', 'View', 'Tools', 'Multiplayer', 'Help'):
    mb.addMenu(n)
win.setMenuBar(mb)
central = QWidget()
lay = QHBoxLayout(central)
tree = QTreeWidget()
tree.setObjectName('GameList')
tree.setHeaderLabels(['Name', 'File type', 'Size', 'Play time'])
tree.setAlternatingRowColors(True)
for i, n in enumerate(('Hyrule Warriors Age of Imprisonment', 'Ocarina of Time Remake', 'Pokemon Pokopia',
                       'Pokemon Legends Z-A', "Let's Go Eevee")):
    it = QTreeWidgetItem([n, 'Directory', f'{20 + i}.0 MB', '26:51:48'])
    tree.addTopLevelItem(it)
tree.setCurrentItem(tree.topLevelItem(0))
tree.setColumnWidth(0, 330)
lay.addWidget(tree, 3)
side = QVBoxLayout()
g = QGroupBox('CPU')
gl = QVBoxLayout(g)
gl.addWidget(QCheckBox('Use multicore'))
c = QCheckBox('Fast GPU time')
c.setChecked(True)
gl.addWidget(c)
cb = QComboBox()
cb.addItems(['6 cores', '4 cores'])
gl.addWidget(cb)
gl.addWidget(QLineEdit('C:/Games'))
s = QSlider(Qt.Horizontal)
s.setValue(60)
gl.addWidget(s)
side.addWidget(g)
tabs = QTabWidget()
for n in ('General', 'Graphics', 'Audio'):
    tabs.addTab(QLabel(f'  {n} settings'), n)
side.addWidget(tabs)
p = QProgressBar()
p.setValue(65)
side.addWidget(p)
row = QHBoxLayout()
row.addWidget(QPushButton('OK'))
row.addWidget(QPushButton('Cancel'))
side.addLayout(row)
lay.addLayout(side, 2)
win.setCentralWidget(central)
sb = QStatusBar()
sb.showMessage('Ready')
win.setStatusBar(sb)
win.show()
app.processEvents()
win.grab().save(out)
print('warnings:', len(msgs))
for m in msgs[:20]:
    print(' ', m)
