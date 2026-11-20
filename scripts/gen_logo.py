from pathlib import Path

def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def quant(c, steps):
    out = []
    for v in c:
        i = min(steps - 1, int(v / 256 * steps))
        lo, hi = (i * 256) // steps, ((i + 1) * 256) // steps
        out.append(lo + (hi - lo) // 2)
    return tuple(out)

# ---------------- moon ----------------
H, W = 14, 30            # rows, cols (px: x=col, y=row*2+half)
cx, cy, r = 15, 14, 14   # main disc
dx, r2 = 8, 15           # shadow disc (crescent on the left)

BRIGHT_LO = (147, 197, 253)
BRIGHT_HI = (240, 248, 255)
SHADOW = (82, 96, 128)
STAR = (210, 225, 250)

def inside(x, y, cxx, cyy, rr):
    return (x - cxx) ** 2 + (y - cyy) ** 2 <= rr * rr

moon_rows = []
for row in range(H):
    cells = []  # (char, color) per column
    for x in range(W):
        tb = inside(x, 2*row, cx, cy, r) and not inside(x, 2*row, cx+dx, cy, r2)
        bb = inside(x, 2*row+1, cx, cy, r) and not inside(x, 2*row+1, cx+dx, cy, r2)
        ts = inside(x, 2*row, cx, cy, r) and inside(x, 2*row, cx+dx, cy, r2)
        bs = inside(x, 2*row+1, cx, cy, r) and inside(x, 2*row+1, cx+dx, cy, r2)
        if tb and bb:
            t = x / (W - 1)
            cells.append(('\u2588', quant(lerp(BRIGHT_HI, BRIGHT_LO, t), 6)))
        elif tb and not bb:
            cells.append(('\u2580', quant(lerp(BRIGHT_HI, BRIGHT_LO, x / (W - 1)), 6)))
        elif bb and not tb:
            cells.append(('\u2584', quant(lerp(BRIGHT_HI, BRIGHT_LO, x / (W - 1)), 6)))
        elif ts or bs:
            cells.append(('\u2588' if (ts and bs) else ('\u2580' if ts else '\u2584'), SHADOW))
        else:
            cells.append((' ', None))
    moon_rows.append(cells)

stars = {(W - 3, 2), (2, H - 3), (W - 7, H - 5)}
for (sx, sy) in stars:
    if moon_rows[sy][sx] == (' ', None):
        moon_rows[sy][sx] = ('*', STAR)

# ---------------- text ----------------
FONT = {
's': [" ####", "#    ", " ### ", "    #", "#### "],
'e': [" ####", "#   #", "#### ", "#    ", " ####"],
'l': ["#  ", "#  ", "#  ", "#  ", "## "],
'n': [" ##  #", "##   #", "##  ##", "##  ##", "##  ##"],
'i': [" # ", "   ", " # ", " # ", "###"],
'c': [" ####", "#    ", "#    ", "#    ", " ####"],
'o': [" ### ", "#   #", "#   #", "#   #", " ### "],
'd': ["    #", " ####", "#  ##", "#   #", " ####"],
}
word = "selenicode"
text_rows = []
for r_ in range(5):
    line = ""
    for ch in word:
        line += FONT[ch][r_] + " "
    text_rows.append(line.rstrip())
TW = max(len(l) for l in text_rows)

# ---------------- emit with SGR runs ----------------
def sgr(c):
    return f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m"

def emit(cells_or_chars, color_at):
    # cells: list of (char, color-or-None); merge consecutive same-color non-space runs
    out = []
    i = 0
    n = len(cells_or_chars)
    while i < n:
        ch, col = cells_or_chars[i]
        if ch == ' ' or col is None:
            j = i
            while j < n and (cells_or_chars[j][0] == ' ' or cells_or_chars[j][1] is None):
                j += 1
            out.append(''.join(c for c, _ in cells_or_chars[i:j]))
            i = j
        else:
            j = i + 1
            while j < n and cells_or_chars[j][0] != ' ' and cells_or_chars[j][1] == col:
                j += 1
            out.append(sgr(col) + ''.join(c for c, _ in cells_or_chars[i:j]) + "\x1b[39m")
            i = j
    return ''.join(out)

lines = []
for cells in moon_rows:
    lines.append(emit(cells, None))
lines.append("")
for r_ in range(5):
    line = text_rows[r_]
    cells = []
    for x, ch in enumerate(line.ljust(TW)):
        if ch == ' ':
            cells.append((' ', None))
        else:
            t = x / (TW - 1)
            cells.append((ch, quant(lerp(BRIGHT_LO, BRIGHT_HI, t), 8)))
    lines.append(emit(cells, None))

data = "\n".join(lines) + "\n"
out_path = Path(__file__).resolve().parent.parent / "assets" / "logo.txt"
out_path.write_text(data, encoding="utf-8")

# preview (stripped)
import re
def preview(cells_list):
    out = []
    for cells in cells_list:
        row = []
        for ch, col in cells:
            if ch == ' ':
                row.append(' ')
            elif col is None or col == SHADOW:
                row.append('.')
            else:
                row.append('#' if ch != '*' else '*')
        out.append("".join(row))
    return out

print("--- brightness preview (#=bright .=shadow) ---")
for i, row in enumerate(preview(moon_rows)):
    print("%2d %s" % (i, row))
print("--- text ---")
for i, ln in enumerate(text_rows):
    print("%2d %s" % (i, ln))
print()
ncols = set()
for m in __import__('re').finditer(r'\x1b\[38;2;(\d+);(\d+);(\d+)m', data):
    ncols.add(m.group(0))
print('distinct colors:', len(ncols))
