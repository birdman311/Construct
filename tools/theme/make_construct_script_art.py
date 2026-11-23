# SPDX-FileCopyrightText: 2026 birdman311
# SPDX-License-Identifier: MIT
# Construct script art in the style of the Zonai seals (brushed oracle-bone-like strokes, rings of
# running text, mint-teal glow, a heavy spiral curl). The glyph shapes are drawn here from stroke
# templates - an original 26-letter cipher alphabet (like Sheikah script is for English) - so the
# rings spell real words. Nothing is traced or taken from any game. Writes sigil, friezes and a key.
import math
import pathlib
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
MINT = (92, 232, 176)
MINT_DEEP = (40, 160, 128)
GOLD = (214, 172, 82)
SS = 3

# ---------------- alphabet: each letter = 2-4 brush strokes in a unit cell (x 0..1, y 0..1) ----------------
# stroke primitives, chosen and jittered per letter with a fixed seed so the alphabet never changes
PRIMS = {
    'zig': [(0.1, 0.25), (0.45, 0.1), (0.3, 0.45), (0.8, 0.3)],
    'hook': [(0.2, 0.15), (0.2, 0.75), (0.55, 0.85), (0.7, 0.65)],
    'em': [(0.05, 0.7), (0.25, 0.3), (0.45, 0.65), (0.65, 0.3), (0.85, 0.7)],
    'tee': [(0.15, 0.2), (0.85, 0.2)],
    'stem': [(0.5, 0.2), (0.5, 0.9)],
    'slash': [(0.25, 0.85), (0.75, 0.15)],
    'tick': [(0.55, 0.45), (0.8, 0.6)],
    'wave': [(0.1, 0.55), (0.3, 0.45), (0.5, 0.6), (0.7, 0.45), (0.9, 0.55)],
    'cup': [(0.2, 0.35), (0.3, 0.75), (0.7, 0.75), (0.8, 0.35)],
    'fork': [(0.5, 0.9), (0.5, 0.5), (0.25, 0.15)],
    'fork2': [(0.5, 0.5), (0.78, 0.15)],
    'curl': [(0.75, 0.3), (0.45, 0.2), (0.3, 0.45), (0.5, 0.65), (0.65, 0.5)],
    'bar': [(0.15, 0.85), (0.85, 0.85)],
    'step': [(0.15, 0.7), (0.4, 0.7), (0.4, 0.4), (0.7, 0.4), (0.7, 0.15)],
}
DOT = 'dot'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def make_alphabet():
    names = list(PRIMS)
    rnd = random.Random(20261122)
    used = set()
    alpha = {}
    for ch in LETTERS:
        while True:
            k = rnd.choice((2, 2, 3, 3, 3))
            combo = tuple(sorted(rnd.sample(names, k)))
            if combo not in used:
                used.add(combo)
                break
        strokes = []
        for n in combo:
            dx, dy = rnd.uniform(-0.08, 0.08), rnd.uniform(-0.08, 0.08)
            flip = rnd.random() < 0.35
            pts = [((1 - x if flip else x) + dx, y + dy) for x, y in PRIMS[n]]
            strokes.append(pts)
        if rnd.random() < 0.35:
            strokes.append([(rnd.uniform(0.2, 0.8), rnd.uniform(0.15, 0.85))])  # dot
        alpha[ch] = strokes
    alpha[' '] = []
    alpha['-'] = [[(0.3, 0.5), (0.7, 0.5)]]
    return alpha


ALPHA = make_alphabet()


def brush(draw, pts, width, color):
    """Brush stroke: stamped discs along the path, thick in the middle, tapered at both ends."""
    if len(pts) == 1:
        x, y = pts[0]
        r = width * 0.75
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        return
    seg = []
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        seg.append((a, b, L))
        total += L
    s = 0.0
    step = max(0.5, width * 0.25)
    for a, b, L in seg:
        n = max(1, int(L / step))
        for i in range(n + 1):
            t = i / n
            u = (s + t * L) / total if total else 0
            r = width * 0.5 * (0.45 + 0.55 * math.sin(math.pi * min(1, max(0, u))) ** 0.6)
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        s += L


def draw_glyph(draw, ch, x, y, size, color, width, angle=0.0, wobble=None):
    """Glyph `ch` with its cell centred at (x, y), rotated by angle (radians, for ring text)."""
    ca, sa = math.cos(angle), math.sin(angle)
    rnd = wobble or random.Random(0)
    for stroke in ALPHA.get(ch, []):
        pts = []
        for px, py in stroke:
            lx = (px - 0.5) * size + rnd.uniform(-0.03, 0.03) * size
            ly = (py - 0.5) * size + rnd.uniform(-0.03, 0.03) * size
            pts.append((x + lx * ca - ly * sa, y + lx * sa + ly * ca))
        brush(draw, pts, width, color)


def glow(img, color, radius, strength=1.0):
    a = img.split()[3].filter(ImageFilter.GaussianBlur(radius)).point(lambda v: min(255, int(v * strength)))
    g = Image.new('RGBA', img.size, color + (0,))
    g.putalpha(a)
    return Image.alpha_composite(g, img)


def ring_text(draw, cx, cy, radius, text, size, color, width, seed, wave=0.0):
    """Write text clockwise around a circle, glyphs upright to the outside, slight wavy baseline."""
    rnd = random.Random(seed)
    n = len(text)
    for i, ch in enumerate(text):
        a = 2 * math.pi * i / n - math.pi / 2
        r = radius + math.sin(i * 0.9) * wave
        draw_glyph(draw, ch, cx + r * math.cos(a), cy + r * math.sin(a), size, color, width,
                   angle=a + math.pi / 2, wobble=rnd)


def fit(text, radius, size, gap=1.25):
    """Repeat text (with separators) to fill a ring of this radius."""
    slots = max(8, int(2 * math.pi * radius / (size * gap)))
    unit = text + ' - '
    s = (unit * (slots // len(unit) + 2))[:slots]
    return s


def spiral_curl(draw, cx, cy, R, width, color):
    """Heavy spiral curl: a thick hook that winds inward, tapering - drawn from an Archimedean spiral."""
    pts = []
    turns = 1.35
    for i in range(0, 241):
        t = i / 240
        a = -math.pi / 2 + t * turns * 2 * math.pi
        r = R * (1 - 0.78 * t)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    # thick body: stamp discs whose size shrinks toward the centre
    for i, (x, y) in enumerate(pts):
        t = i / (len(pts) - 1)
        rr = width * (1.25 - 0.9 * t)
        draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=color)
    # tail flick out of the top
    x0, y0 = pts[0]
    brush(draw, [(x0, y0), (x0 - R * 0.25, y0 - R * 0.18), (x0 - R * 0.42, y0 - R * 0.05)], width * 1.1, color)


# ---------------- background sigil ----------------
def sigil(size):
    S = size * SS
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    mint = MINT + (235,)
    deep = MINT_DEEP + (210,)
    lw = int(S * 0.006)
    # three rings of running script, like a seal
    rings = [(0.455, 'CONSTRUCT SWITCH EMULATOR', 0.034, 11), (0.395, 'AGE OF IMPRISONMENT', 0.031, 23),
             (0.34, 'POWER OF THE ANCIENT CONSTRUCTS', 0.027, 37)]
    for rr, text, gs, seed in rings:
        R = S * rr
        ring_text(d, c, c, R, fit(text, R, S * gs), S * gs, mint, lw * 0.9, seed, wave=S * 0.002)
    # separating lines between rings (thin, slightly broken like a carving)
    for rr in (0.488, 0.425, 0.367, 0.312):
        R = S * rr
        for k in range(36):
            a0 = k * 10 + 1.5
            d.arc((c - R, c - R, c + R, c + R), a0, a0 + 7.5, fill=deep, width=max(1, lw // 2))
    # inner disc: a heavy spiral curl in the middle, four small curls at the quarters
    spiral_curl(d, c, c, S * 0.17, S * 0.022, mint)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        x, y = c + S * 0.255 * math.cos(a), c + S * 0.255 * math.sin(a)
        spiral_curl(d, x, y, S * 0.035, S * 0.006, deep)
    # four short vertical script columns, like the side panels of a seal
    for k, word in enumerate(('ZONAI', 'GUIDE', 'LIGHT', 'STONE')):
        a = k * math.pi / 2
        bx, by = c + S * 0.24 * math.cos(a), c + S * 0.24 * math.sin(a)
        for j, ch in enumerate(word):
            off = (j - (len(word) - 1) / 2) * S * 0.022
            # columns run tangentially
            x = bx - off * math.sin(a)
            y = by + off * math.cos(a)
            draw_glyph(d, ch, x, y, S * 0.019, deep, lw * 0.9, angle=a, wobble=random.Random(k * 7 + j))
    comp = glow(img, MINT, 5 * SS, 0.9)
    small = comp.resize((size, size), Image.LANCZOS)
    small.putalpha(small.split()[3].point(lambda v: int(v * 0.22)))
    return small


# ---------------- inscription band ----------------
def frieze(w, h, text, seed, lead_px):
    W, H = w * SS, h * SS
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    L = lead_px * SS
    g = H * 0.56
    lw = H * 0.06
    step = g * 1.18
    rnd = random.Random(seed)
    unit = text + ' - '
    x = L + step
    i = 0
    while x < W - step * 0.5:
        fade = max(0.0, min(1.0, (x - L) / (W * 0.06)))
        ch = unit[i % len(unit)]
        y = H / 2 + math.sin(i * 0.7) * H * 0.04
        draw_glyph(d, ch, x, y, g, MINT + (int(235 * fade),), lw, wobble=rnd)
        x += step
        i += 1
    # thin broken guide lines above and below
    for yy, a in ((H * 0.08, 120), (H * 0.92, 90)):
        xx = L
        while xx < W:
            fade = max(0.0, min(1.0, (xx - L) / (W * 0.06)))
            d.line((xx, yy, xx + step * 2.2, yy), fill=MINT_DEEP + (int(a * fade),), width=max(1, SS))
            xx += step * 2.6
    return glow(img, MINT, 2 * SS, 1.0).resize((w, h), Image.LANCZOS)


# glyph atlas for Construct's scrolling ribbons (glyph_ribbon.cpp): one 128px cell per character in
# ATLAS order, drawn white with a white glow so the app can tint them (mint normally, gold for secrets).
ATLAS = LETTERS + '-*'
CELL = 128
atlas = Image.new('RGBA', (CELL * len(ATLAS), CELL), (0, 0, 0, 0))
for idx, ch in enumerate(ATLAS):
    layer = Image.new('RGBA', (CELL * SS, CELL * SS), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    c = CELL * SS / 2
    if ch == '*':
        spiral_curl(ld, c, c * 1.04, CELL * SS * 0.24, CELL * SS * 0.035, (255, 255, 255, 255))
    else:
        draw_glyph(ld, ch, c, c, CELL * SS * 0.62, (255, 255, 255, 255), CELL * SS * 0.075,
                   wobble=random.Random(idx))
    cell = glow(layer, (255, 255, 255), 3 * SS, 0.9).resize((CELL, CELL), Image.LANCZOS)
    atlas.alpha_composite(cell, (idx * CELL, 0))
atlas.save(out / 'glyphs.png')

# key: the alphabet with Latin letters under each glyph
cols, cell = 9, 120
key = Image.new('RGBA', (cols * cell, 3 * (cell + 40) + 20), (11, 18, 20, 255))
kd = ImageDraw.Draw(key)
try:
    font = ImageFont.truetype('pala.ttf', 26)
except OSError:
    font = ImageFont.load_default()
for idx, ch in enumerate(LETTERS):
    cx = (idx % cols) * cell + cell / 2
    cy = (idx // cols) * (cell + 40) + cell / 2 + 10
    layer = Image.new('RGBA', key.size, (0, 0, 0, 0))
    draw_glyph(ImageDraw.Draw(layer), ch, cx, cy, cell * 0.62, MINT + (255,), cell * 0.06)
    key.alpha_composite(glow(layer, MINT, 4, 0.8))
    kd.text((cx - 8, cy + cell * 0.42), ch, fill=(214, 172, 82, 255), font=font)
key.save(out / '_alphabet_key.png')

print('script art written to', out)
