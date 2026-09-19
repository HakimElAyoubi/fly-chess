"""Phase 6: the garden, the chess set, and the fly, as a MuJoCo scene.

flybody works in centimetres and grams, and its fly is real size: 3.4 mm long and 1 mg. So the
chess set is built to the fly rather than the other way round. A square is 3.2 mm, about the
size of the fly itself, the whole board is 2.6 cm across, and the tallest piece is under 2 mm.
It sits on a mossy stone in a lawn, with flowers and grass towering over the game, trees on the
skyline and the sun low in the sky.

Nothing here is physically simulated. The fly and the pieces are placed directly each frame,
which is the whole point: the only computation in the demo is the brain choosing a move.

    python -m flybrain.scene --out data/scene.png       # render the opening position
"""
import argparse
import math
from pathlib import Path
import numpy as np
import chess

FLYBODY = Path("/private/tmp/claude-501/-Users-hakim-Desktop-fruit-fly/87f30919-7987-456d-9e6d-4dae2ee87d86/scratchpad/menagerie/flybody")

SQUARE = 0.32            # cm, about one fly long
BOARD = 8 * SQUARE
THICK = 0.025            # board slab
RIM = 0.10               # border around the playing surface
STONE_R = BOARD * 0.80   # the slab the board rests on (kept for callers; the stone itself is wider now)
STONE_RX, STONE_RY = BOARD * 1.10, BOARD * 0.95
GROUND = -0.30           # lawn level; the stone top is z = 0
PIECE_SCALE = 1.15
NOCOL = 'contype="0" conaffinity="0" group="1"'


# ----------------------------------------------------------------------------- the chess set
def piece_geoms(kind):
    """One piece as a stack of primitives: (type, size, z centre[, euler]) in cm. Every piece
    stands on a felt disc, then a turned base with a collar, then its own body."""
    base = [("cylinder", (0.056, 0.004), 0.004, None, "felt"),
            ("cylinder", (0.054, 0.010), 0.016), ("cylinder", (0.046, 0.006), 0.031),
            ("cylinder", (0.038, 0.006), 0.041)]
    if kind == chess.PAWN:
        return base + [("capsule", (0.019, 0.016), 0.066), ("cylinder", (0.030, 0.005), 0.088),
                       ("sphere", (0.028,), 0.114)]
    if kind == chess.ROOK:
        return base + [("cylinder", (0.032, 0.030), 0.076), ("cylinder", (0.042, 0.006), 0.110),
                       ("cylinder", (0.040, 0.012), 0.126),
                       ("box", (0.040, 0.011, 0.012), 0.148), ("box", (0.011, 0.040, 0.012), 0.148),
                       ("cylinder", (0.026, 0.006, ), 0.146, None, "shadowwell")]
    if kind == chess.KNIGHT:
        return base + [("capsule", (0.026, 0.020), 0.064),
                       ("capsule", (0.022, 0.034), 0.118, (0.55, 0, 0)),          # neck, leaning forward
                       ("ellipsoid", (0.020, 0.036, 0.018), 0.164, (0.35, 0, 0)),  # head
                       ("box", (0.006, 0.006, 0.014), 0.186, (0.2, 0, 0.3), None, (0.012, -0.006)),
                       ("box", (0.006, 0.006, 0.014), 0.186, (0.2, 0, -0.3), None, (-0.012, -0.006)),
                       ("capsule", (0.008, 0.026), 0.146, (0.9, 0, 0), None, (0.0, -0.020))]  # mane
    if kind == chess.BISHOP:
        return base + [("capsule", (0.023, 0.026), 0.072), ("cylinder", (0.032, 0.005), 0.104),
                       ("ellipsoid", (0.030, 0.030, 0.044), 0.150),
                       ("box", (0.004, 0.034, 0.030), 0.166, (0, 0, 0.6), "obsidian_slit"),
                       ("sphere", (0.012,), 0.204)]
    if kind == chess.QUEEN:
        return base + [("capsule", (0.027, 0.034), 0.080), ("cylinder", (0.036, 0.005), 0.120),
                       ("ellipsoid", (0.038, 0.038, 0.038), 0.156), ("cylinder", (0.044, 0.006), 0.196),
                       ("cylinder", (0.034, 0.008), 0.210)] + \
               [("sphere", (0.009,), 0.226, None, None, (0.028 * math.cos(a), 0.028 * math.sin(a)))
                for a in np.linspace(0, 2 * math.pi, 7)[:-1]] + [("sphere", (0.014,), 0.240)]
    if kind == chess.KING:
        return base + [("capsule", (0.027, 0.038), 0.084), ("cylinder", (0.036, 0.005), 0.128),
                       ("ellipsoid", (0.036, 0.036, 0.040), 0.166), ("cylinder", (0.040, 0.006), 0.208),
                       ("cylinder", (0.016, 0.010), 0.224),
                       ("box", (0.007, 0.007, 0.032), 0.262), ("box", (0.024, 0.007, 0.007), 0.262)]
    raise ValueError(kind)


def square_xy(sq):
    """Centre of a chess square in scene coordinates, with a1 at the near-left."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return ((f - 3.5) * SQUARE, (r - 3.5) * SQUARE)


def piece_body(name, kind, colour, pos):
    mat = "ivory" if colour == chess.WHITE else "ebony"
    g = []
    for spec in piece_geoms(kind):
        typ, size, z = spec[0], spec[1], spec[2]
        euler = spec[3] if len(spec) > 3 else None
        m = spec[4] if len(spec) > 4 and spec[4] else mat
        if m == "obsidian_slit": m = "ebony" if colour == chess.WHITE else "ivory_slit"
        if m == "shadowwell": m = "ebony" if colour == chess.WHITE else "ebony_deep"
        off = spec[5] if len(spec) > 5 and spec[5] else (0.0, 0.0)
        e = f' euler="{euler[0]} {euler[1]} {euler[2]}"' if euler else ""
        s = " ".join(f"{v * PIECE_SCALE:.4f}" for v in size)
        g.append(f'      <geom type="{typ}" size="{s}" pos="{off[0]:.4f} {off[1]:.4f} {z * PIECE_SCALE:.4f}"{e} '
                 f'material="{m}" {NOCOL}/>')
    return (f'    <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">\n'
            f'      <freejoint name="{name}_free"/>\n' + "\n".join(g) + "\n    </body>")


def board_geoms():
    """The board: a stepped walnut frame with a maple inlay line, and sixty-four squares."""
    g = []
    half = BOARD / 2
    # two-step frame: a wider, lower plinth and the rim proper
    g.append(f'    <geom name="plinth" type="box" size="{half + RIM + 0.05:.3f} {half + RIM + 0.05:.3f} {THICK * 0.8:.4f}" '
             f'pos="0 0 {THICK * 0.8:.4f}" material="walnut_dark" {NOCOL}/>')
    for nm, sx, sy, px, py in (("rim_n", half + RIM, RIM / 2, 0, half + RIM / 2), ("rim_s", half + RIM, RIM / 2, 0, -(half + RIM / 2)),
                               ("rim_e", RIM / 2, half, half + RIM / 2, 0), ("rim_w", RIM / 2, half, -(half + RIM / 2), 0)):
        g.append(f'    <geom name="{nm}" type="box" size="{sx:.3f} {sy:.3f} {THICK + 0.004:.4f}" pos="{px:.3f} {py:.3f} {THICK + 0.004:.4f}" '
                 f'material="walnut" {NOCOL}/>')
    # the inlay: a thin pale line let into the rim around the playing surface
    ins = 0.012
    for nm, sx, sy, px, py in (("in_n", half + ins, 0.004, 0, half + ins), ("in_s", half + ins, 0.004, 0, -(half + ins)),
                               ("in_e", 0.004, half + ins, half + ins, 0), ("in_w", 0.004, half + ins, -(half + ins), 0)):
        g.append(f'    <geom name="{nm}" type="box" size="{sx:.3f} {sy:.3f} {THICK + 0.0045:.4f}" pos="{px:.3f} {py:.3f} {THICK + 0.004:.4f}" '
                 f'material="maple_inlay" {NOCOL}/>')
    for sq in chess.SQUARES:
        x, y = square_xy(sq)
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        g.append(f'    <geom name="sq{sq}" type="box" size="{SQUARE/2:.4f} {SQUARE/2:.4f} {THICK:.4f}" '
                 f'pos="{x:.4f} {y:.4f} {THICK:.4f}" material="{"maple" if light else "walnut_sq"}" {NOCOL}/>')
    return g


# ----------------------------------------------------------------------------- the garden
def stone():
    """A flattened, slightly irregular stone with moss on its shoulders."""
    g = [f'    <geom name="slab" type="ellipsoid" size="{STONE_RX:.3f} {STONE_RY:.3f} 0.42" pos="0 0 -0.42" material="stone" {NOCOL}/>',
         f'    <geom name="slab2" type="ellipsoid" size="{STONE_RX * 0.86:.3f} {STONE_RY * 0.92:.3f} 0.36" pos="0.25 -0.10 -0.36" euler="0 0 0.4" material="stone2" {NOCOL}/>',
         f'    <geom name="slabtop" type="cylinder" size="{STONE_RX * 0.93:.3f} 0.012" pos="0 0 -0.012" material="stone" {NOCOL}/>']
    rng = np.random.default_rng(11)
    for i in range(14):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(STONE_RX * 0.62, STONE_RX * 0.98)
        x, y = r * math.cos(a), r * math.sin(a) * (STONE_RY / STONE_RX)
        if abs(x) < BOARD / 2 + RIM + 0.15 and abs(y) < BOARD / 2 + RIM + 0.15: continue
        s = rng.uniform(0.12, 0.34)
        g.append(f'    <geom name="moss{i}" type="ellipsoid" size="{s:.3f} {s * rng.uniform(0.6, 1.0):.3f} {s * 0.16:.3f}" '
                 f'pos="{x:.3f} {y:.3f} {-0.004:.3f}" euler="0 0 {rng.uniform(0, 3.1):.2f}" material="moss{rng.integers(1, 3)}" {NOCOL}/>')
    return g


def grass_blades(rng, n_tufts=380, per_tuft=(3, 8), inner=None, outer=BOARD * 8.0):
    """Flat, tapered, arching blades in tufts, all clear of the stone, taller further out."""
    inner = inner if inner is not None else STONE_RX * 1.12
    out = []; i = 0
    for _ in range(n_tufts):
        a = rng.uniform(0, 2 * math.pi)
        r = inner + (outer - inner) * math.sqrt(rng.uniform(0, 1))
        cx, cy = r * math.cos(a), r * math.sin(a)
        for _ in range(int(rng.integers(*per_tuft))):
            x = cx + rng.normal(0, 0.07); y = cy + rng.normal(0, 0.07)
            if math.hypot(x / STONE_RX, y / STONE_RY) < 1.08:          # never on the stone
                continue
            near = math.hypot(x, y) / outer
            h = rng.uniform(0.5, 1.6) + 2.8 * near * rng.uniform(0.4, 1.0)
            w = rng.uniform(0.012, 0.030)
            arch = rng.uniform(0.12, 0.62)
            outward = math.atan2(y, x)
            spread = 0.5 if math.hypot(x, y) < inner * 1.9 else 2.4
            face = outward + rng.uniform(-spread, spread)
            mat = f"grass{rng.integers(1, 6)}"
            segs, z, tilt, lean = [], 0.0, 0.0, 0.0
            for k, frac in enumerate((0.42, 0.33, 0.25)):
                L = h * frac
                tilt += arch * (0.45 + 0.8 * k)
                segs.append(f'      <geom type="box" size="{w * (1 - 0.32 * k):.4f} 0.0022 {L/2:.3f}" '
                            f'pos="{lean + math.sin(tilt) * L / 2:.3f} 0 {z + math.cos(tilt) * L / 2:.3f}" '
                            f'euler="0 {tilt:.3f} 0" material="{mat}" {NOCOL}/>')
                lean += math.sin(tilt) * L; z += math.cos(tilt) * L
            out.append(f'    <body name="blade{i}" pos="{x:.3f} {y:.3f} {GROUND}" euler="0 0 {face:.3f}">\n'
                       + "\n".join(segs) + "\n    </body>")
            i += 1
    return out


def flowers(rng, n=44):
    """Daisies, buttercups and a few poppies: a stem, a ring of petal ellipsoids, a centre."""
    out = []
    kinds = [("daisy", "petal_white", "centre_yellow", 6, 0.16, 0.8),
             ("buttercup", "petal_yellow", "centre_amber", 5, 0.11, 0.6),
             ("poppy", "petal_red", "centre_dark", 4, 0.20, 1.0),
             ("cornflower", "petal_blue", "centre_dark", 6, 0.12, 0.7)]
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(STONE_RX * 1.35, BOARD * 6.5)
        x, y = r * math.cos(a), r * math.sin(a)
        kind, petal, centre, npet, pr, wt = kinds[int(rng.choice(4, p=[0.42, 0.25, 0.15, 0.18]))]
        h = rng.uniform(1.0, 2.8) * (0.7 + 0.6 * r / (BOARD * 6.5))
        lean = rng.uniform(-0.18, 0.18); face = rng.uniform(0, 2 * math.pi)
        g = [f'      <geom type="capsule" size="0.026 {h/2:.3f}" pos="0 0 {h/2:.3f}" material="stem" {NOCOL}/>',
             f'      <geom type="ellipsoid" size="0.16 0.05 0.008" pos="0.13 0 {h * 0.40:.3f}" euler="0 -0.5 0" material="stem" {NOCOL}/>',
             f'      <geom type="ellipsoid" size="0.13 0.045 0.008" pos="-0.10 0.03 {h * 0.58:.3f}" euler="0 0.55 2.6" material="stem" {NOCOL}/>']
        for k in range(npet):
            b = 2 * math.pi * k / npet
            g.append(f'      <geom type="ellipsoid" size="{pr:.3f} {pr * 0.42:.3f} 0.012" '
                     f'pos="{pr * 0.95 * math.cos(b):.3f} {pr * 0.95 * math.sin(b):.3f} {h:.3f}" euler="0 0.25 {b:.3f}" '
                     f'material="{petal}" {NOCOL}/>')
        g.append(f'      <geom type="sphere" size="{pr * 0.34:.3f}" pos="0 0 {h + 0.015:.3f}" material="{centre}" {NOCOL}/>')
        out.append(f'    <body name="flower{i}" pos="{x:.3f} {y:.3f} {GROUND}" euler="{lean:.2f} {lean * 0.6:.2f} {face:.2f}">\n'
                   + "\n".join(g) + "\n    </body>")
    return out


def mushrooms(rng, n=4):
    out = []
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(BOARD * 2.4, BOARD * 4.5)
        x, y = r * math.cos(a), r * math.sin(a)
        h = rng.uniform(0.35, 0.8); cap = rng.uniform(0.22, 0.42)
        out.append(f'    <body name="shroom{i}" pos="{x:.3f} {y:.3f} {GROUND}" euler="0 0 {rng.uniform(0, 3):.2f}">\n'
                   f'      <geom type="capsule" size="{cap * 0.22:.3f} {h/2:.3f}" pos="0 0 {h/2:.3f}" material="petal_white" {NOCOL}/>\n'
                   f'      <geom type="ellipsoid" size="{cap:.3f} {cap * 0.92:.3f} {cap * 0.55:.3f}" pos="0 0 {h:.3f}" material="mushroom" {NOCOL}/>\n'
                   f'      <geom type="sphere" size="{cap * 0.14:.3f}" pos="{cap * 0.4:.3f} {cap * 0.2:.3f} {h + cap * 0.45:.3f}" material="petal_white" {NOCOL}/>\n'
                   f'      <geom type="sphere" size="{cap * 0.10:.3f}" pos="{-cap * 0.3:.3f} {-cap * 0.35:.3f} {h + cap * 0.40:.3f}" material="petal_white" {NOCOL}/>\n'
                   '    </body>')
    return out


HAZE = np.array([0.80, 0.84, 0.90])


def hazed(rgb, r, near=350.0, far=1000.0, strength=0.72):
    """Aerial perspective by hand: blend a colour toward the sky haze with distance."""
    t = min(1.0, max(0.0, (r - near) / (far - near))) * strength
    c = np.array(rgb) * (1 - t) + HAZE * t
    return f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"


CANOPY = [(0.22, 0.42, 0.18), (0.30, 0.52, 0.22), (0.40, 0.58, 0.24)]
BARK = (0.34, 0.25, 0.17)


def trees(rng):
    """Trees on the skyline: a trunk, a few limbs, and a crown of overlapping canopies. Placed
    in a wide ring so that every camera angle has some, the deepest ones hazed toward the sky."""
    out = []
    # azimuth 2.51 rad is kept clear: that is where the sun sits
    spots = [(520, 0.35, 70), (700, 1.05, 105), (600, 1.75, 88), (820, 3.30, 120), (560, 3.05, 82),
             (760, 3.70, 108), (640, 4.40, 92), (500, 5.10, 66), (880, 5.75, 130), (960, 0.75, 140), (900, 2.05, 128), (940, 4.05, 135),
             (430, 2.95, 58), (450, 4.90, 62), (470, 1.40, 60), (540, 3.40, 74), (620, 5.50, 84), (680, 2.82, 96), (760, 2.20, 100)]
    for i, (r, a, h) in enumerate(spots):
        x, y = r * math.cos(a), r * math.sin(a)
        tr = h * 0.055
        crown = h * 0.62
        bark = hazed(BARK, r)
        g = [f'      <geom type="capsule" size="{tr:.2f} {h * 0.30:.2f}" pos="0 0 {h * 0.30:.2f}" rgba="{bark}" {NOCOL}/>']
        for k in range(3):
            b = rng.uniform(0, 2 * math.pi); t = rng.uniform(0.5, 0.9)
            g.append(f'      <geom type="capsule" size="{tr * 0.45:.2f} {h * 0.16:.2f}" pos="{h * 0.12 * math.cos(b):.2f} {h * 0.12 * math.sin(b):.2f} {crown * 0.85:.2f}" '
                     f'euler="{-t * math.sin(b):.2f} {t * math.cos(b):.2f} 0" rgba="{bark}" {NOCOL}/>')
        for k in range(9):
            b = rng.uniform(0, 2 * math.pi); d = rng.uniform(0, 0.55) * h * 0.42
            cz = crown + rng.uniform(-0.12, 0.30) * h
            s = h * rng.uniform(0.20, 0.34)
            g.append(f'      <geom type="ellipsoid" size="{s:.2f} {s * rng.uniform(0.85, 1.1):.2f} {s * rng.uniform(0.7, 0.9):.2f}" '
                     f'pos="{d * math.cos(b):.2f} {d * math.sin(b):.2f} {cz:.2f}" euler="0 0 {b:.2f}" rgba="{hazed(CANOPY[rng.integers(3)], r)}" {NOCOL}/>')
        out.append(f'    <body name="tree{i}" pos="{x:.2f} {y:.2f} {GROUND}">\n' + "\n".join(g) + "\n    </body>")
    return out


def bushes(rng, n=22):
    out = []
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(120.0, 380.0)
        x, y = r * math.cos(a), r * math.sin(a)
        s = rng.uniform(6.0, 14.0) * (0.5 + 0.5 * r / 380.0)
        g = []
        for k in range(6):
            b = rng.uniform(0, 2 * math.pi); d = rng.uniform(0, 0.6) * s
            g.append(f'      <geom type="ellipsoid" size="{s * rng.uniform(0.5, 0.8):.2f} {s * rng.uniform(0.5, 0.8):.2f} {s * rng.uniform(0.35, 0.55):.2f}" '
                     f'pos="{d * math.cos(b):.2f} {d * math.sin(b):.2f} {s * rng.uniform(0.2, 0.45):.2f}" rgba="{hazed(CANOPY[rng.integers(3)], r, 150, 500, 0.5)}" {NOCOL}/>')
        out.append(f'    <body name="bush{i}" pos="{x:.2f} {y:.2f} {GROUND}">\n' + "\n".join(g) + "\n    </body>")
    return out


def sun_and_clouds(rng):
    """The sun as a glowing disc with a soft halo, and a few clouds, all far out."""
    sx, sy, sz = -730.0, 530.0, 105.0                        # low over the far side of the board, in view
    g = [f'    <geom name="sun" type="sphere" size="17" pos="{sx} {sy} {sz}" material="sun" {NOCOL}/>',
         f'    <geom name="halo1" type="sphere" size="27" pos="{sx} {sy} {sz}" material="halo1" {NOCOL}/>',
         f'    <geom name="halo2" type="sphere" size="44" pos="{sx} {sy} {sz}" material="halo2" {NOCOL}/>']
    # clouds at set bearings so that the default view gets a few between the trees
    for i, (a, r, el) in enumerate([(1.72, 900, 0.20), (1.96, 1000, 0.16), (2.14, 950, 0.23), (2.86, 1050, 0.15), (3.10, 900, 0.19),
                                    (0.6, 950, 0.18), (3.9, 900, 0.20), (4.8, 1000, 0.17), (5.6, 950, 0.22)]):
        z = r * el
        cx, cy = r * math.cos(a), r * math.sin(a)
        w = rng.uniform(70, 150)
        for k in range(5):
            d = rng.uniform(-0.8, 0.8) * w; dz = rng.uniform(-0.15, 0.25) * w
            g.append(f'    <geom type="ellipsoid" size="{w * rng.uniform(0.45, 0.7):.1f} {w * rng.uniform(0.35, 0.55):.1f} {w * rng.uniform(0.18, 0.3):.1f}" '
                     f'pos="{cx + d * math.cos(a + 1.57):.1f} {cy + d * math.sin(a + 1.57):.1f} {z + dz:.1f}" material="cloud" {NOCOL}/>')
    return g


def pebbles(rng, n=40):
    out = []
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(STONE_RX * 1.15, STONE_RX * 3.2)
        s = rng.uniform(0.02, 0.09)
        out.append(f'    <geom name="peb{i}" type="ellipsoid" size="{s:.3f} {s*rng.uniform(0.6,1.0):.3f} {s*0.55:.3f}" '
                   f'pos="{r*math.cos(a):.3f} {r*math.sin(a):.3f} {GROUND + s*0.45:.3f}" '
                   f'euler="0 0 {rng.uniform(0,3.14):.2f}" material="pebble{rng.integers(1, 3)}" {NOCOL}/>')
    return out


def dewdrops(rng, n=40):
    out = []
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi)
        on_stone = rng.random() < 0.5
        r = rng.uniform(BOARD * 0.70, STONE_RX * 0.95) if on_stone else rng.uniform(STONE_RX * 1.15, STONE_RX * 2.6)
        x, y = r * math.cos(a), r * math.sin(a) * (STONE_RY / STONE_RX if on_stone else 1.0)
        if on_stone and abs(x) < BOARD / 2 + RIM + 0.08 and abs(y) < BOARD / 2 + RIM + 0.08: continue
        s = rng.uniform(0.008, 0.024)
        z = 0.002 + s * 0.6 if on_stone else GROUND + s * 0.6
        out.append(f'    <geom name="dew{i}" type="ellipsoid" size="{s:.4f} {s:.4f} {s*0.7:.4f}" '
                   f'pos="{x:.3f} {y:.3f} {z:.3f}" material="dew" {NOCOL}/>')
    return out


# ----------------------------------------------------------------------------- the scene
def build(board=None, seed=3):
    """Return the scene XML for a position (default: the opening)."""
    board = board or chess.Board()
    rng = np.random.default_rng(seed)
    pieces = []
    for sq, pc in board.piece_map().items():
        x, y = square_xy(sq)
        pieces.append(piece_body(f"p{sq}", pc.piece_type, pc.color, (x, y, 2 * THICK)))
    for i in range(4):                       # parked bodies for promotions
        pieces.append(piece_body(f"spare{i}", chess.QUEEN, chess.WHITE if i < 2 else chess.BLACK,
                                 (BOARD * 1.6 + 0.2 * i, -BOARD, 2 * THICK)))
    NL = chr(10)
    return f"""<mujoco model="fly chess garden">
  <compiler angle="radian"/>
  <include file="fruitfly_nolights.xml"/>

  <visual>
    <global offwidth="3840" offheight="2160" azimuth="132" elevation="-14"/>
    <quality shadowsize="16384" offsamples="16"/>
    <headlight ambient="0.26 0.27 0.30" diffuse="0.14 0.14 0.15" specular="0.03 0.03 0.03"/>
    <map stiffness="1e+04" stiffnessrot="5e+04" force="2e-05" shadowclip="2.4" shadowscale="1.2" znear="0.002" zfar="1000"/>
    <scale jointwidth="0.004" framewidth="0.004"/>
  </visual>

  <statistic meansize="0.05" extent="4" center="0 0 0.3"/>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.22 0.42 0.78" rgb2="0.93 0.88 0.80" width="1024" height="1024"/>
    <texture name="lawntex" type="2d" builtin="flat" rgb1="0.30 0.47 0.20" rgb2="0.22 0.38 0.16" width="800" height="800"
      mark="random" markrgb="0.40 0.56 0.24" random="0.6"/>
    <material name="lawn" texture="lawntex" texrepeat="60 60" texuniform="true" specular="0.04" shininess="0.05"/>
    <texture name="stonetex" type="2d" builtin="flat" rgb1="0.46 0.44 0.40" rgb2="0.38 0.37 0.34" width="600" height="600"
      mark="random" markrgb="0.55 0.52 0.47" random="0.30"/>
    <material name="stone" texture="stonetex" texrepeat="6 6" texuniform="true" specular="0.10" shininess="0.12" reflectance="0.01"/>
    <material name="stone2" texture="stonetex" texrepeat="4 4" texuniform="true" rgba="0.9 0.88 0.86 1" specular="0.10" shininess="0.12"/>
    <material name="moss1" rgba="0.36 0.52 0.22 1" specular="0.05" shininess="0.1"/>
    <material name="moss2" rgba="0.44 0.60 0.26 1" specular="0.05" shininess="0.1"/>
    <material name="pebble1" rgba="0.50 0.47 0.43 1" specular="0.2" shininess="0.3"/>
    <material name="pebble2" rgba="0.62 0.58 0.52 1" specular="0.2" shininess="0.3"/>
    <texture name="mapletex" type="2d" builtin="flat" rgb1="0.90 0.84 0.70" rgb2="0.84 0.77 0.62" width="400" height="400"
      mark="random" markrgb="0.94 0.89 0.77" random="0.25"/>
    <texture name="walnuttex" type="2d" builtin="flat" rgb1="0.36 0.24 0.15" rgb2="0.29 0.19 0.12" width="400" height="400"
      mark="random" markrgb="0.42 0.29 0.18" random="0.25"/>
    <material name="maple"       texture="mapletex"  texrepeat="3 3" texuniform="true" specular="0.45" shininess="0.70" reflectance="0.10"/>
    <material name="walnut_sq"   texture="walnuttex" texrepeat="3 3" texuniform="true" specular="0.45" shininess="0.70" reflectance="0.10"/>
    <material name="walnut"      texture="walnuttex" texrepeat="8 8" texuniform="true" specular="0.35" shininess="0.55" reflectance="0.05"/>
    <material name="walnut_dark" rgba="0.24 0.15 0.09 1" specular="0.3" shininess="0.5"/>
    <material name="maple_inlay" rgba="0.93 0.88 0.74 1" specular="0.4" shininess="0.6"/>
    <material name="ivory"       rgba="0.93 0.89 0.80 1" specular="0.40" shininess="0.65" reflectance="0.05"/>
    <material name="ivory_slit"  rgba="0.80 0.75 0.66 1" specular="0.40" shininess="0.65"/>
    <material name="ebony"       rgba="0.11 0.09 0.09 1" specular="0.60" shininess="0.90" reflectance="0.10"/>
    <material name="ebony_deep"  rgba="0.05 0.04 0.04 1" specular="0.2" shininess="0.5"/>
    <material name="felt"        rgba="0.45 0.12 0.12 1" specular="0.02" shininess="0.02"/>
    <material name="grass1"   rgba="0.30 0.54 0.22 1" specular="0.2" shininess="0.3"/>
    <material name="grass2"   rgba="0.38 0.62 0.26 1" specular="0.2" shininess="0.3"/>
    <material name="grass3"   rgba="0.24 0.46 0.20 1" specular="0.2" shininess="0.3"/>
    <material name="grass4"   rgba="0.46 0.68 0.30 1" specular="0.25" shininess="0.35"/>
    <material name="grass5"   rgba="0.56 0.70 0.30 1" specular="0.25" shininess="0.35"/>
    <material name="stem"     rgba="0.34 0.55 0.24 1" specular="0.15" shininess="0.2"/>
    <material name="petal_white"  rgba="0.97 0.96 0.92 1" specular="0.2" shininess="0.3"/>
    <material name="petal_yellow" rgba="0.98 0.82 0.20 1" specular="0.3" shininess="0.4"/>
    <material name="petal_red"    rgba="0.86 0.16 0.12 1" specular="0.3" shininess="0.4"/>
    <material name="petal_blue"   rgba="0.36 0.46 0.86 1" specular="0.3" shininess="0.4"/>
    <material name="centre_yellow" rgba="0.98 0.78 0.16 1" specular="0.3" shininess="0.3"/>
    <material name="centre_amber"  rgba="0.80 0.52 0.10 1" specular="0.3" shininess="0.3"/>
    <material name="centre_dark"   rgba="0.18 0.12 0.10 1" specular="0.3" shininess="0.3"/>
    <material name="mushroom" rgba="0.78 0.22 0.16 1" specular="0.3" shininess="0.4"/>
    <material name="bark"     rgba="0.34 0.25 0.17 1" specular="0.05" shininess="0.05"/>
    <material name="canopy1"  rgba="0.22 0.42 0.18 1" specular="0.08" shininess="0.1"/>
    <material name="canopy2"  rgba="0.30 0.52 0.22 1" specular="0.08" shininess="0.1"/>
    <material name="canopy3"  rgba="0.40 0.58 0.24 1" specular="0.08" shininess="0.1"/>
    <material name="sun"   rgba="1.00 0.96 0.82 1" emission="1.0" specular="0" shininess="0"/>
    <material name="halo1" rgba="1.00 0.92 0.70 0.35" emission="1.0" specular="0" shininess="0"/>
    <material name="halo2" rgba="1.00 0.90 0.66 0.14" emission="1.0" specular="0" shininess="0"/>
    <material name="cloud" rgba="0.97 0.97 1.00 0.97" emission="0.42" specular="0" shininess="0"/>
    <material name="dew"   rgba="0.85 0.93 0.98 0.60" specular="0.9" shininess="0.95" reflectance="0.3"/>
  </asset>

  <worldbody>
    <light name="sunlight" directional="true" pos="-8 6 5" dir="0.74 -0.54 -0.40" diffuse="1.00 0.86 0.66"
      specular="0.36 0.32 0.24" castshadow="true"/>
    <light name="skylight" directional="true" pos="2 6 9" dir="-0.2 -0.6 -1" diffuse="0.24 0.29 0.38" castshadow="false"/>
    <light name="bounce" directional="true" pos="6 -6 2" dir="-0.6 0.6 -0.35" diffuse="0.34 0.30 0.22" castshadow="false"/>
    <light name="fill" directional="true" pos="3 -8 4" dir="-0.3 0.8 -0.5" diffuse="0.26 0.26 0.28" castshadow="false"/>
    <geom name="ground" type="plane" size="1500 1500 0.2" pos="0 0 {GROUND}" material="lawn"/>
{NL.join(stone())}
{NL.join(board_geoms())}
{NL.join(pieces)}
{NL.join(grass_blades(rng))}
{NL.join(flowers(rng))}
{NL.join(mushrooms(rng))}
{NL.join(pebbles(rng))}
{NL.join(dewdrops(rng))}
{NL.join(bushes(rng))}
{NL.join(trees(rng))}
{NL.join(sun_and_clouds(rng))}
  </worldbody>
</mujoco>
"""


def write(path=None, board=None, seed=3):
    path = Path(path or (FLYBODY / "fly_chess_scene.xml"))
    nolights = path.parent / "fruitfly_nolights.xml"
    if not nolights.exists():                 # flybody's three tracking lights darken every far geom
        src = (path.parent / "fruitfly.xml").read_text().splitlines(keepends=True)
        nolights.write_text("".join(l for l in src if "<light " not in l))
    path.write_text(build(board, seed))
    return path


def still(out, width=1600, height=1000, seed=3, azimuth=124, elevation=-12, distance=3.9, lookat=(-0.05, -0.10, 0.14)):
    import mujoco
    from PIL import Image
    p = write(seed=seed)
    m = mujoco.MjModel.from_xml_path(str(p)); d = mujoco.MjData(m)
    fly = m.joint("free").qposadr[0]
    d.qpos[fly:fly + 7] = [-(BOARD / 2 + RIM + 0.30), -(BOARD / 2 + RIM + 0.22), 2 * THICK + 0.1235, 0.707, 0, 0, 0.707]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(out)
    return m.ngeom


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scene.png")
    ap.add_argument("--width", type=int, default=1600); ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--azimuth", type=float, default=124); ap.add_argument("--elevation", type=float, default=-12)
    ap.add_argument("--distance", type=float, default=3.9)
    a = ap.parse_args()
    n = still(a.out, a.width, a.height, a.seed, a.azimuth, a.elevation, a.distance)
    print(f"wrote {a.out}: {n} geoms (board {BOARD:.2f} cm across, square {SQUARE*10:.1f} mm, fly 3.4 mm)")
