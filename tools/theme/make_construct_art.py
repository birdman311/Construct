# SPDX-FileCopyrightText: 2026 birdman311
# SPDX-License-Identifier: MIT
# Original decorative art for Construct's theme: an invented "ancient" glyph alphabet, six original
# emblems, a large background sigil, inscription friezes for the bars, and gold frame corners.
# Everything is drawn from geometry here; nothing is taken from any game.
import math
import pathlib
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

GOLD = (214, 172, 82)
GOLD_BRIGHT = (240, 206, 120)
BRONZE = (176, 112, 58)
JADE = (47, 191, 143)
TEAL = (70, 227, 191)
EMBER = (214, 96, 64)
AZURE = (88, 160, 224)
VIOLET = (150, 120, 210)
PARCH = (236, 226, 198)
SS = 4  # supersampling


def rgba(c, a):
    return (c[0], c[1], c[2], int(a))


# ---------------- glyph alphabet ----------------
def glyph(draw, cx, cy, r, seed, color, width):
    """One invented glyph: a ring or half ring with radial strokes, dots and small arcs."""
    rnd = random.Random(seed)
    kind = rnd.randrange(4)
    box = (cx - r, cy - r, cx + r, cy + r)
    if kind == 0:
        draw.ellipse(box, outline=color, width=width)
    elif kind == 1:
        start = rnd.choice((0, 90, 180, 270))
        draw.arc(box, start, start + 180, fill=color, width=width)
    elif kind == 2:
        draw.ellipse(box, outline=color, width=width)
        ri = r * 0.45
        draw.ellipse((cx - ri, cy - ri, cx + ri, cy + ri), outline=color, width=width)
    else:
        draw.arc(box, 30, 330, fill=color, width=width)
    for _ in range(rnd.randrange(1, 4)):
        a = math.radians(rnd.choice(range(0, 360, 45)))
        r0 = r * rnd.choice((0.0, 0.3, 0.5))
        r1 = r * rnd.choice((1.0, 1.25))
        draw.line((cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + r1 * math.cos(a), cy + r1 * math.sin(a)),
                  fill=color, width=width)
    if rnd.random() < 0.7:
        d = max(width, r * 0.16)
        ox, oy = rnd.choice(((0, 0), (0, -r * 0.5), (r * 0.5, 0), (0, r * 0.5)))
        draw.ellipse((cx + ox - d, cy + oy - d, cx + ox + d, cy + oy + d), fill=color)
    if rnd.random() < 0.4:
        a = rnd.choice((0, 90, 180, 270))
        ra = r * 0.6
        draw.arc((cx - ra, cy - ra, cx + ra, cy + ra), a, a + 90, fill=color, width=width)


def glow_layer(img, color, radius, strength=1.0):
    """Coloured glow behind the shapes of img."""
    alpha = img.split()[3].filter(ImageFilter.GaussianBlur(radius))
    alpha = alpha.point(lambda v: min(255, int(v * strength)))
    g = Image.new('RGBA', img.size, color + (0,))
    g.putalpha(alpha)
    return g


def finish(img, size):
    return img.resize(size, Image.LANCZOS)


# ---------------- emblems (original) ----------------
def emblem(draw, cx, cy, r, which, color, width):
    def poly(points):
        draw.line(points + [points[0]], fill=color, width=width, joint='curve')
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=width)
    ri = r * 0.62
    if which == 'flame':
        pts = []
        for i in range(0, 361, 10):
            t = math.radians(i)
            rr = ri * (0.55 + 0.45 * abs(math.sin(t * 1.5)))
            pts.append((cx + rr * math.sin(t) * 0.7, cy - rr * math.cos(t) * (1.1 if math.cos(t) > 0 else 0.6)))
        draw.line(pts, fill=color, width=width)
    elif which == 'wave':
        for k in (-1, 0, 1):
            pts = [(cx - ri + i, cy + k * ri * 0.45 + math.sin(i / ri * 2 * math.pi) * ri * 0.18)
                   for i in range(0, int(2 * ri) + 1, max(1, int(ri / 12)))]
            draw.line(pts, fill=color, width=width)
    elif which == 'leaf':
        draw.arc((cx - ri * 1.2, cy - ri, cx + ri * 0.4, cy + ri), -60, 60, fill=color, width=width)
        draw.arc((cx - ri * 0.4, cy - ri, cx + ri * 1.2, cy + ri), 120, 240, fill=color, width=width)
        draw.line((cx, cy - ri * 0.85, cx, cy + ri), fill=color, width=width)
    elif which == 'mountain':
        poly([(cx - ri, cy + ri * 0.6), (cx - ri * 0.25, cy - ri * 0.6), (cx + ri * 0.15, cy),
              (cx + ri * 0.45, cy - ri * 0.3), (cx + ri, cy + ri * 0.6)])
    elif which == 'bolt':
        draw.line([(cx + ri * 0.3, cy - ri), (cx - ri * 0.35, cy + ri * 0.05), (cx + ri * 0.25, cy + ri * 0.05),
                   (cx - ri * 0.3, cy + ri)], fill=color, width=width, joint='curve')
    elif which == 'star':
        pts = []
        for i in range(16):
            a = math.radians(i * 22.5 - 90)
            rr = ri if i % 2 == 0 else ri * 0.42
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        poly(pts)
    d = max(width, r * 0.07)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, y = cx + r * 1.18 * math.cos(a), cy + r * 1.18 * math.sin(a)
        draw.ellipse((x - d, y - d, x + d, y + d), fill=color)


EMBLEMS = [('flame', EMBER), ('wave', AZURE), ('leaf', JADE), ('mountain', BRONZE), ('bolt', VIOLET),
           ('star', GOLD_BRIGHT)]


# ---------------- background sigil ----------------
def sigil(size):
    S = size * SS
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    w = 2 * SS
    for rr, col, a in ((0.48, GOLD, 200), (0.455, GOLD, 120), (0.33, TEAL, 170), (0.315, GOLD, 120),
                       (0.17, GOLD, 180), (0.08, TEAL, 200)):
        r = S * rr
        d.ellipse((c - r, c - r, c + r, c + r), outline=rgba(col, a), width=w)
    # glyph ring
    n = 28
    for i in range(n):
        a = 2 * math.pi * i / n - math.pi / 2
        r = S * 0.392
        glyph(d, c + r * math.cos(a), c + r * math.sin(a), S * 0.024, 1000 + i, rgba(GOLD, 200), w)
    # 12 rays between inner rings, alternating colours
    for i in range(12):
        a = 2 * math.pi * i / 12 + math.pi / 12  # between the medallions, never through them
        r0, r1 = S * 0.18, S * (0.31 if i % 2 else 0.27)
        d.line((c + r0 * math.cos(a), c + r0 * math.sin(a), c + r1 * math.cos(a), c + r1 * math.sin(a)),
               fill=rgba(TEAL if i % 2 else GOLD, 190), width=w)
    # six emblem medallions on the inner ring
    for i, (name, col) in enumerate(EMBLEMS):
        a = 2 * math.pi * i / 6 - math.pi / 2
        r = S * 0.245
        emblem(d, c + r * math.cos(a), c + r * math.sin(a), S * 0.045, name, rgba(col, 220), w)
    # hexagon (not a triangle) linking the medallions
    pts = [(c + S * 0.245 * math.cos(2 * math.pi * i / 6 - math.pi / 2),
            c + S * 0.245 * math.sin(2 * math.pi * i / 6 - math.pi / 2)) for i in range(6)]
    for i in range(6):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 6]
        t0, t1 = 0.3, 0.7
        d.line((x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1),
               fill=rgba(GOLD, 110), width=w)
    # centre: 8-point compass and core
    for i in range(8):
        a = 2 * math.pi * i / 8
        r1 = S * (0.14 if i % 2 == 0 else 0.1)
        d.line((c, c, c + r1 * math.cos(a), c + r1 * math.sin(a)), fill=rgba(GOLD_BRIGHT, 170), width=w)
    r = S * 0.03
    d.ellipse((c - r, c - r, c + r, c + r), fill=rgba(TEAL, 230))
    # tick marks on the outer ring
    for i in range(120):
        a = 2 * math.pi * i / 120
        r0, r1 = S * 0.455, S * (0.47 if i % 5 else 0.48)
        d.line((c + r0 * math.cos(a), c + r0 * math.sin(a), c + r1 * math.cos(a), c + r1 * math.sin(a)),
               fill=rgba(GOLD, 150), width=SS)
    glow = glow_layer(img, TEAL, 6 * SS, 0.8)
    comp = Image.alpha_composite(glow, img)
    small = finish(comp, (size, size))
    # faint: it sits behind the game list
    a = small.split()[3].point(lambda v: int(v * 0.16))
    small.putalpha(a)
    return small


# ---------------- inscription friezes ----------------
def frieze(w, h, seed, emblems=False, lead=0.0):
    """Inscription band. The first `lead` (fraction of the width) stays empty for menu names; the band
    fades in after it. Anchored at the left in the theme (right/bottom anchoring is unreliable)."""
    W, H = w * SS, h * SS
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = max(1, SS)
    cy = H / 2
    L = W * lead
    span = W - L

    def fade(x):
        return max(0.0, min(1.0, (x - L) / (span * 0.12)))
    # fading gold rules top and bottom
    for y in (H * 0.12, H * 0.88):
        for x in range(int(L), W, SS):
            a = 255 * fade(x) * (0.55 if y < cy else 0.4)
            d.line((x, y, x + SS, y), fill=rgba(GOLD, a), width=lw)
    r = H * 0.26
    step = r * 3.0
    x = W - step * 0.7
    i = 0
    while x > L + step * 0.5:
        t = fade(x) / 2.2
        if emblems and i % 7 == 3:
            name, col = EMBLEMS[(i // 7) % len(EMBLEMS)]
            emblem(d, x, cy, r * 1.05, name, rgba(col, 235 * min(1, t * 2.2)), lw)
        elif i % 9 == 0:
            s = r * 0.55
            d.polygon([(x, cy - s), (x + s, cy), (x, cy + s), (x - s, cy)], outline=rgba(TEAL, 230 * min(1, t * 2.2)))
        else:
            glyph(d, x, cy, r, seed + i, rgba(GOLD if i % 4 else BRONZE, 230 * min(1, t * 2.2)), lw)
        x -= step
        i += 1
    glow = glow_layer(img, TEAL, 2.5 * SS, 0.9)
    return Image.alpha_composite(glow, img)


# ---------------- 9-slice frame with ornamented corners ----------------
def frame(size, corner):
    S, C = size * SS, corner * SS
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = SS
    m = SS * 1.5
    # thin edges
    d.line((C, m, S - C, m), fill=rgba(GOLD, 170), width=lw)
    d.line((C, S - m, S - C, S - m), fill=rgba(GOLD, 120), width=lw)
    d.line((m, C, m, S - C), fill=rgba(GOLD, 140), width=lw)
    d.line((S - m, C, S - m, S - C), fill=rgba(GOLD, 140), width=lw)
    for fx, fy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        def P(x, y):
            return (S - x if fx else x, S - y if fy else y)
        # L bracket
        d.line([P(m, C), P(m, m), P(C, m)], fill=rgba(GOLD_BRIGHT, 235), width=lw * 2, joint='curve')
        # inner curl
        a0 = {(0, 0): 180, (1, 0): 270, (0, 1): 90, (1, 1): 0}[(fx, fy)]
        cx, cy = P(m + C * 0.45, m + C * 0.45)
        rr = C * 0.32
        d.arc((cx - rr, cy - rr, cx + rr, cy + rr), a0, a0 + 90, fill=rgba(GOLD, 220), width=lw)
        # teal jewel
        jx, jy = P(m + C * 0.2, m + C * 0.2)
        s = C * 0.12
        d.polygon([(jx, jy - s), (jx + s, jy), (jx, jy + s), (jx - s, jy)], fill=rgba(TEAL, 240))
    return finish(Image.alpha_composite(glow_layer(img, TEAL, 1.5 * SS, 0.5), img), (size, size))


def save_pair(make, name, *args):
    """name.png at 1x and name@2x.png at 2x."""
    make(*args, scale=1).save(out / f'{name}.png')
    make(*args, scale=2).save(out / f'{name}@2x.png')


sigil(520).save(out / 'sigil.png')
sigil(1040).save(out / 'sigil@2x.png')
# bars: 2600 logical px wide, empty for the first 440 px (menu names), inscriptions after that
for name, (w, h, seed, em, lead) in {'frieze_top': (2600, 28, 11, False, 520 / 2600),
                                     'frieze_bottom': (2600, 26, 77, True, 300 / 2600),
                                     'frieze_header': (420, 30, 303, False, 0.0)}.items():
    frieze(w, h, seed, em, lead).resize((w, h), Image.LANCZOS).save(out / f'{name}.png')
    frieze(w * 2, h * 2, seed, em, lead).resize((w * 2, h * 2), Image.LANCZOS).save(out / f'{name}@2x.png')
frame(48, 16).save(out / 'frame.png')
frame(96, 32).save(out / 'frame@2x.png')

# preview sheet for checking by eye
sheet = Image.new('RGBA', (1200, 820), (11, 18, 20, 255))
sg = Image.open(out / 'sigil.png')
a = sg.split()[3].point(lambda v: min(255, v * 5))
sg.putalpha(a)
sheet.alpha_composite(sg, (20, 20))
y = 40
for n in ('frieze_top', 'frieze_bottom', 'frieze_header'):
    sheet.alpha_composite(Image.open(out / f'{n}@2x.png').resize((640, 34)), (560, y))
    y += 70
fr = Image.open(out / 'frame@2x.png').resize((192, 192), Image.NEAREST)
sheet.alpha_composite(fr, (580, 300))
sheet.save(out / '_preview.png')
print('art written to', out)
