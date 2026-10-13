#!/usr/bin/env python3
"""Schematic of the measures on and around the cortical surface (CAT website), as SVG.

A cortical ribbon (gray matter) with its central surface runs through four groups:
cortical surfaces | cortical measures | intensity-/projection-based measures | folding measures.
Light and dark colors switch with prefers-color-scheme inside the SVG, so one file
serves both themes. No dependencies.

  python3 tools/surface_schematic.py assets/img/surface-schematic.svg
"""
import math
import sys

W, H = 1100, 318
T = 36                       # ribbon (GM) thickness
Y0 = 140                     # central surface of the flat ribbon
PIAL, WHITE = Y0 - T / 2, Y0 + T / 2
SKULL = PIAL - 16
YF = 262                     # centre of the fundus arcs
RS = 24                      # central-surface radius at a fundus (sulci)
RG = 46                      # central-surface radius at a crown (gyri) and the shoulders
LY = 74                      # baseline of the measure labels (two lines)
COLS = [(0, 176), (176, 400), (400, 630), (630, W)]   # x ranges of the four groups
X_RIBBON = 152               # the ribbon starts right of the surface labels
S0 = 384                     # deep sulcus between groups 2 and 3
XM = 642                     # start of the meander in group 4

# light / dark colors: one per item, used for its marks and (darker or lighter) for its label
COLORS = {
    'text': ('#17202b', '#e6ebf0'), 'muted': ('#556170', '#9aa6b2'),
    'band': ('rgba(23,32,43,.045)', 'rgba(230,235,240,.05)'),
    'gm': ('#d5dce4', '#3a4552'), 'gmvol': ('#9aa6b2', '#6a7684'), 'skull': ('#8b96a3', '#6f7b88'),
    'csf-fill': ('rgba(42,120,214,.15)', 'rgba(121,184,255,.17)'), 'lesion-fill': ('#dccfa9', '#7d7158'),
    'pial': ('#0b64a0', '#4cc3ff'), 'pial-line': ('#1899dc', '#4cc3ff'),
    'central': ('#8f5400', '#ffb43c'), 'central-line': ('#e08a00', '#ffb43c'),
    'white': ('#1f7028', '#5fe356'), 'white-line': ('#2ea836', '#5fe356'),
    'thick': ('#5a3fd0', '#a99bff'), 'area': ('#b23a0a', '#ff8a5c'), 'csf': ('#1f63b0', '#79b8ff'),
    'fmri': ('#b02a62', '#ff8fbf'), 'smri': ('#1f63b0', '#79b8ff'), 'lesion': ('#7a6331', '#cbb98d'),
    'dmri': ('#097052', '#3ddc9f'), 'gyri': ('#c42f2f', '#ff7b7a'), 'sulci': ('#1f63b0', '#79b8ff'),
    'depth': ('#096a7c', '#3fd0e0'), 'arearel': ('#25702f', '#69db7c'),
}


def style():
    light = ' '.join(f'--{k}:{v[0]};' for k, v in COLORS.items())
    dark = ' '.join(f'--{k}:{v[1]};' for k, v in COLORS.items())
    return f'''
svg {{ {light} }}
@media (prefers-color-scheme: dark) {{ svg {{ {dark} }} }}
text {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 14px; fill: var(--text); }}
.h {{ font-size: 15px; font-weight: 650; }}
.n {{ font-size: 12.5px; }}
.s {{ font-size: 11.5px; }}
'''


out = []
el = out.append


def f(x):
    return f'{x:.2f}'.rstrip('0').rstrip('.')


class Path:
    """Central surface as straight segments and circular arcs, tracking the current point."""

    def __init__(self, x, y):
        self.x, self.y, self.d = x, y, [f'M{f(x)},{f(y)}']

    def h(self, x):
        self.d.append(f'H{f(x)}')
        self.x = x

    def v(self, y):
        self.d.append(f'V{f(y)}')
        self.y = y

    def arc(self, r, x, y, sweep):
        self.d.append(f'A{f(r)},{f(r)} 0 0 {sweep} {f(x)},{f(y)}')
        self.x, self.y = x, y

    def down(self):                  # shoulder: heading right -> heading down
        self.arc(RG, self.x + RG, self.y + RG, 1)

    def fundus(self):                # down the bank, around the fundus, heading up
        self.v(YF)
        self.arc(RS, self.x + 2 * RS, YF, 0)

    def crown(self):                 # up the bank, over the crown, heading down
        self.v(Y0 + RG)
        self.arc(RG, self.x + 2 * RG, Y0 + RG, 1)

    def up(self):                    # up the bank, shoulder: heading right
        self.v(Y0 + RG)
        self.arc(RG, self.x + RG, Y0, 1)

    def __str__(self):
        return ' '.join(self.d)


def line(x1, y1, x2, y2, color, w=2.2, extra=''):
    el(f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" style="stroke:var(--{color})" stroke-width="{w}"{extra}/>')


def arrow(x1, y1, x2, y2, color, both=False, w=2.2, head=7):
    """Line with a filled head at (x2, y2), and at (x1, y1) if both."""
    a = math.atan2(y2 - y1, x2 - x1)
    ux, uy = math.cos(a), math.sin(a)

    def tip(x, y, sx, sy):
        bx, by = x - head * sx, y - head * sy
        px, py = -sy * head * 0.52, sx * head * 0.52
        el(f'<path d="M{f(x)},{f(y)} L{f(bx + px)},{f(by + py)} L{f(bx - px)},{f(by - py)} Z" style="fill:var(--{color})"/>')
    line(x1 + (head * ux if both else 0), y1 + (head * uy if both else 0), x2 - head * ux, y2 - head * uy, color, w)
    tip(x2, y2, ux, uy)
    if both:
        tip(x1, y1, -ux, -uy)


def dot(x, y, color, r=4.2):
    el(f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" style="fill:var(--{color})"/>')


def text(x, y, lines, color='text', anchor='middle', cls=None):
    lines = [lines] if isinstance(lines, str) else lines
    spans = ''.join(f'<tspan x="{f(x)}" dy="{0 if i == 0 else 16}">{s}</tspan>' for i, s in enumerate(lines))
    c = f' class="{cls}"' if cls else ''
    el(f'<text{c} x="{f(x)}" y="{f(y)}" text-anchor="{anchor}" style="fill:var(--{color})">{spans}</text>')


def build():
    el(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">')
    el('<title id="t">Measures on and around the cortical surface</title>')
    el('<desc id="d">Schematic cortical ribbon with the skull, pial, central and white matter surfaces; cortical measures '
       '(gray matter thickness, area and volume, CSF and white matter volume); intensity- and projection-based measures '
       'sampled from fMRI, structural and diffusion MRI, including a white matter lesion; and folding measures (curvature '
       'as one over the local radius of gyri and sulci, sulcal depth to the convex hull, and the area relation of the '
       'folded and the unfolded surface).</desc>')
    el(f'<style>{style()}</style>')
    for i in (1, 3):                                       # bands behind groups 2 and 4
        x0, x1 = COLS[i]
        el(f'<rect x="{x0}" y="0" width="{x1 - x0}" height="{H}" rx="10" style="fill:var(--band)"/>')

    # central surface: flat, deep sulcus S0, flat, meander S1 G1 S2 G2 S3, flat
    p = Path(X_RIBBON, Y0)
    p.h(S0 - RS - RG)
    p.down(); p.fundus(); p.up()
    p.h(XM)
    p.down(); p.fundus(); p.crown(); p.fundus(); p.crown(); p.fundus(); p.up()
    p.h(W)
    central = str(p)
    s1 = XM + RG + RS
    g1, s2 = s1 + RS + RG, s1 + 2 * (RS + RG)
    g2, s3 = g1 + 2 * (RS + RG), s2 + 2 * (RS + RG)

    # CSF in group 2: between skull and pial surface, and inside the deep sulcus
    gap = RS - T / 2
    el(f'<path d="M{COLS[1][0]},{f(SKULL)} H{COLS[1][1]} V{f(PIAL)} H{COLS[1][0]} Z" style="fill:var(--csf-fill)"/>')
    el(f'<path d="M{f(S0 - gap)},{f(PIAL)} V{f(YF)} A{f(gap)},{f(gap)} 0 0 0 {f(S0 + gap)},{f(YF)} V{f(PIAL)} Z" style="fill:var(--csf-fill)"/>')
    line(X_RIBBON, SKULL, W, SKULL, 'skull', 1.6, ' stroke-dasharray="1.6 3.2"')
    el(f'<path d="{central}" fill="none" style="stroke:var(--gm)" stroke-width="{T}"/>')

    gx0, gx1 = 252, 304                                     # GM volume block (group 2), under the lines
    el(f'<rect x="{gx0}" y="{f(PIAL)}" width="{gx1 - gx0}" height="{T}" style="fill:var(--gmvol)"/>')
    line(X_RIBBON, PIAL, COLS[1][0], PIAL, 'pial-line', 3)
    line(X_RIBBON, WHITE, COLS[1][0], WHITE, 'white-line', 3)
    el(f'<path d="{central}" fill="none" style="stroke:var(--central-line)" stroke-width="2.6"/>')

    # group 1: the four surfaces
    for y, name, color in [(SKULL, 'skull surface', 'muted'), (PIAL, 'pial surface', 'pial'),
                           (Y0, 'central surface', 'central'), (WHITE, 'white matter surface', 'white')]:
        text(X_RIBBON - 10, y + 4.5, name, color, 'end')

    # group 2: cortical measures
    arrow(gx0, PIAL + 1, gx0, WHITE - 1, 'thick', both=True)
    arrow(gx0 + 4, Y0, gx1, Y0, 'area', both=True)
    text(gx0 - 8, LY, ['GM', 'thickness'], 'thick', 'end')
    text(gx0 + 8, LY, ['GM', 'area'], 'area', 'start')
    text(S0 - 30, LY, ['CSF', 'volume'], 'csf')
    text((gx0 + gx1) / 2, WHITE + 24, 'GM volume', 'muted')
    text(278, 246, 'WM volume', 'muted')

    # group 3: intensity- / projection-based measures
    for k in range(4):                                     # fMRI sampled along the central surface
        dot(S0 + RS, 186 + 22 * k, 'fmri')
    text(S0 + RS + 30, LY, ['fMRI', 'data'], 'fmri')
    sx = 494                                               # sMRI: sampling below the surface, lesion projection
    line(sx, Y0, sx, WHITE + 30, 'smri')
    dot(sx, Y0, 'smri')
    dot(sx, WHITE + 30, 'smri')
    el(f'<ellipse cx="{sx + 20}" cy="{f(WHITE + 64)}" rx="17" ry="24" style="fill:var(--lesion-fill)"/>')
    arrow(sx + 20, WHITE + 74, sx + 20, Y0 + 5, 'smri')
    dot(sx + 20, Y0, 'smri')
    text(sx + 10, LY, ['sMRI', 'data'], 'smri')
    text(sx + 20, WHITE + 108, ['WM', 'lesion'], 'lesion')
    dx = 584                                               # dMRI: projection from the white matter
    arrow(dx, WHITE + 66, dx, Y0 + 5, 'dmri')
    dot(dx, Y0, 'dmri')
    text(dx, LY, ['dMRI', 'data'], 'dmri')

    # group 4: folding measures
    for cx, cy, r, color, a0, a1 in [(s1, YF, RS, 'sulci', 0, 180), (g1, Y0 + RG, RG, 'gyri', 180, 360)]:
        el(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="none" style="stroke:var(--{color})" stroke-width="1.8" stroke-dasharray="2 3"/>')
        b0, b1 = math.radians(a0), math.radians(a1)
        el(f'<path d="M{f(cx + r * math.cos(b0))},{f(cy + r * math.sin(b0))} A{f(r)},{f(r)} 0 0 1 '
           f'{f(cx + r * math.cos(b1))},{f(cy + r * math.sin(b1))}" fill="none" style="stroke:var(--{color})" stroke-width="3.2"/>')
    text(g1 - 30, LY, ['curvature', '(1 / local radius)'])
    text(g1, Y0 + RG - 2, ['outward', 'folding', '(gyri)'], 'gyri', cls='n')
    text(g1, YF - 4, ['inward', 'folding', '(sulci)'], 'sulci', cls='n')
    line(g1, Y0, g2, Y0, 'depth', 1.6, ' stroke-dasharray="4 3"')   # convex hull of the central surface
    arrow(s2, YF + RS, s2, Y0, 'depth')
    text(s2, Y0 - 6, 'convex hull', 'depth', cls='s')
    text(s2, LY, ['sulcal', 'depth'], 'depth')
    q = Path(g2, Y0)                                       # folded central surface ...
    q.arc(RG, g2 + RG, Y0 + RG, 1)
    q.fundus()
    q.up()
    el(f'<path d="{q}" fill="none" style="stroke:var(--arearel)" stroke-width="3.2"/>')
    line(g2, SKULL, q.x, SKULL, 'arearel', 3, ' stroke-dasharray="5 3.5"')   # ... and unfolded
    text(s3, LY, ['area relation', '(folded vs. unfolded)'], 'arearel')

    for (x0, x1), lines in zip(COLS, [['Cortical', 'surfaces'], ['Cortical', 'measures'],
                                      ['Intensity- / projection-', 'based measures'],
                                      ['Folding / gyrification / sulcification', 'measures']]):
        text((x0 + x1) / 2, 24, lines, cls='h')
    el('</svg>')
    return '\n'.join(out)


if __name__ == '__main__':
    open(sys.argv[1], 'w').write(build())
