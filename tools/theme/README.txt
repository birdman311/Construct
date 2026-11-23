Construct theme ("Age of Imprisonment style"): original art and stylesheet, no game assets.
  python make_construct_art.py <out_art_dir>                      -> glyph friezes, sigil, emblems, gold frame
  python make_construct_theme_hwai.py <construct_root> <art_dir>   -> dist/qt_themes/construct (style.qss, rc/, style.qrc)
  python qss_preview.py <construct_root>/dist/qt_themes/construct out.png   -> offscreen preview (PySide6)
Then rebuild. Note: Construct's Qt ignores background-position right/bottom; anchor images left/top/center.
  python make_construct_script_art.py <art_dir>   -> (run after make_construct_art.py) replaces sigil + friezes with the brush-script cipher alphabet (Zonai-seal style, original glyphs); writes _alphabet_key.png
