"""Phase 6: the garden, the chess set, and the fly, as a MuJoCo scene.

flybody works in centimetres and grams, and its fly is real size: 3.4 mm long and 1 mg. So the
chess set is built to the fly rather than the other way round. A square is 3.2 mm, about the
size of the fly itself, the whole board is 2.6 cm across, and the tallest piece is under 2 mm.
It sits on a stone in grass that towers over the game.

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
STONE_R = BOARD * 0.80   # the slab the board rests on
PIECE_SCALE = 1.0

# every piece as a stack of primitives: (kind, size, z-offset, [extra])
def piece_geoms(kind):
    """Return a list of (type, size tuple, z centre) for one piece, in cm."""
    base = [("cylinder", (0.052, 0.012), 0.012), ("cylinder", (0.040, 0.008), 0.030)]
    if kind == chess.PAWN:
        return base + [("capsule", (0.020, 0.018), 0.058), ("sphere", (0.030,), 0.098)]
    if kind == chess.ROOK:
        return base + [("cylinder", (0.034, 0.030), 0.068), ("cylinder", (0.044, 0.012), 0.110),
                       ("box", (0.044, 0.010, 0.012), 0.132), ("box", (0.010, 0.044, 0.012), 0.132)]
    if kind == chess.KNIGHT:
        return base + [("capsule", (0.026, 0.024), 0.062), ("capsule", (0.024, 0.030), 0.112, (0.5, 0, 0)),
                       ("box", (0.016, 0.022, 0.012), 0.150)]
    if kind == chess.BISHOP:
        return base + [("capsule", (0.024, 0.026), 0.064), ("ellipsoid", (0.032, 0.032, 0.044), 0.125),
                       ("sphere", (0.014,), 0.172)]
    if kind == chess.QUEEN:
        return base + [("capsule", (0.028, 0.032), 0.070), ("ellipsoid", (0.040, 0.040, 0.038), 0.136),
                       ("cylinder", (0.044, 0.008), 0.176), ("sphere", (0.018,), 0.196)]
    if kind == chess.KING:
        return base + [("capsule", (0.028, 0.036), 0.074), ("ellipsoid", (0.038, 0.038, 0.040), 0.146),
                       ("box", (0.008, 0.008, 0.030), 0.200), ("box", (0.024, 0.008, 0.008), 0.196)]
    raise ValueError(kind)


def square_xy(sq):
    """Centre of a chess square in scene coordinates, with a1 at the near-left."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return ((f - 3.5) * SQUARE, (r - 3.5) * SQUARE)


def piece_body(name, kind, colour, pos):
    mat = "ivory" if colour == chess.WHITE else "obsidian"
    g = []
    for spec in piece_geoms(kind):
        typ, size, z = spec[0], spec[1], spec[2]
        euler = f' euler="{spec[3][0]} {spec[3][1]} {spec[3][2]}"' if len(spec) > 3 else ""
        s = " ".join(f"{v * PIECE_SCALE:.4f}" for v in size)
        g.append(f'      <geom type="{typ}" size="{s}" pos="0 0 {z * PIECE_SCALE:.4f}"{euler} material="{mat}" '
                 f'contype="0" conaffinity="0" group="1"/>')
    return (f'    <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">\n'
            f'      <freejoint name="{name}_free"/>\n' + "\n".join(g) + "\n    </body>")


def grass_blades(rng, n_tufts=110, per_tuft=(3, 9), inner=None, outer=BOARD * 6.0):
    """Flat, tapered, arching blades growing in tufts, all of them clear of the stone."""
    inner = inner if inner is not None else STONE_R * 1.18
    out = []; i = 0
    for _ in range(n_tufts):
        a = rng.uniform(0, 2 * math.pi)
        r = inner + (outer - inner) * math.sqrt(rng.uniform(0, 1))
        cx, cy = r * math.cos(a), r * math.sin(a)
        for _ in range(int(rng.integers(*per_tuft))):
            x = cx + rng.normal(0, 0.06); y = cy + rng.normal(0, 0.06)
            if math.hypot(x, y) < inner * 0.96:          # never on the stone
                continue
            h = rng.uniform(0.6, 3.8)
            w = rng.uniform(0.012, 0.030)
            arch = rng.uniform(0.12, 0.62)
            # a blade leans along its own +x, so face it outward (with jitter) and it can never
            # arch back over the stone; blades further out are free to lean any way they like
            outward = math.atan2(y, x)
            spread = 0.5 if math.hypot(x, y) < inner * 1.9 else 2.4
            face = outward + rng.uniform(-spread, spread)
            mat = f"grass{rng.integers(1, 5)}"
            segs, z, tilt, lean = [], 0.0, 0.0, 0.0
            for k, frac in enumerate((0.42, 0.33, 0.25)):
                L = h * frac
                tilt += arch * (0.45 + 0.8 * k)
                segs.append(f'      <geom type="box" size="{w * (1 - 0.32 * k):.4f} 0.0022 {L/2:.3f}" '
                            f'pos="{lean + math.sin(tilt) * L / 2:.3f} 0 {z + math.cos(tilt) * L / 2:.3f}" '
                            f'euler="0 {tilt:.3f} 0" material="{mat}" contype="0" conaffinity="0" group="1"/>')
                lean += math.sin(tilt) * L; z += math.cos(tilt) * L
            out.append(f'    <body name="blade{i}" pos="{x:.3f} {y:.3f} -0.30" euler="0 0 {face:.3f}">\n'
                       + "\n".join(segs) + "\n    </body>")
            i += 1
    return out


def pebbles(rng, n=34):
    out = []
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(STONE_R * 1.15, STONE_R * 3.0)
        s = rng.uniform(0.02, 0.075)
        out.append(f'    <geom name="peb{i}" type="ellipsoid" size="{s:.3f} {s*rng.uniform(0.6,1.0):.3f} {s*0.55:.3f}" '
                   f'pos="{r*math.cos(a):.3f} {r*math.sin(a):.3f} {-0.30 + s*0.45:.3f}" '
                   f'euler="0 0 {rng.uniform(0,3.14):.2f}" material="pebble" contype="0" conaffinity="0" group="1"/>')
    return out


def dewdrops(rng, n=34):
    out = []
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi)
        on_stone = rng.random() < 0.55
        r = rng.uniform(BOARD * 0.62, STONE_R * 0.97) if on_stone else rng.uniform(STONE_R * 1.1, STONE_R * 2.6)
        s = rng.uniform(0.008, 0.022)
        z = 0.002 + s * 0.6 if on_stone else -0.30 + s * 0.6
        out.append(f'    <geom name="dew{i}" type="ellipsoid" size="{s:.4f} {s:.4f} {s*0.7:.4f}" '
                   f'pos="{r*math.cos(a):.3f} {r*math.sin(a):.3f} {z:.3f}" material="dew" '
                   f'contype="0" conaffinity="0" group="1"/>')
    return out


def build(board=None, seed=3):
    """Return the scene XML for a position (default: the opening)."""
    board = board or chess.Board()
    rng = np.random.default_rng(seed)
    squares = []
    for sq in chess.SQUARES:
        x, y = square_xy(sq)
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        squares.append(f'    <geom name="sq{sq}" type="box" size="{SQUARE/2:.4f} {SQUARE/2:.4f} {THICK:.4f}" '
                       f'pos="{x:.4f} {y:.4f} {THICK:.4f}" material="{"light" if light else "dark"}" '
                       f'contype="0" conaffinity="0" group="1"/>')
    pieces = []
    for sq, pc in board.piece_map().items():
        x, y = square_xy(sq)
        pieces.append(piece_body(f"p{sq}", pc.piece_type, pc.color, (x, y, 2 * THICK)))
    # parked bodies for captured pieces and for any piece that promotion may need later
    for i in range(4):
        pieces.append(piece_body(f"spare{i}", chess.QUEEN, chess.WHITE if i < 2 else chess.BLACK,
                                 (BOARD * 1.6 + 0.2 * i, -BOARD, 2 * THICK)))
    return f"""<mujoco model="fly chess garden">
  <include file="fruitfly.xml"/>

  <visual>
    <global offwidth="2400" offheight="1500" azimuth="132" elevation="-14"/>
    <quality shadowsize="16384" offsamples="32"/>
    <headlight ambient="0.22 0.24 0.28" diffuse="0.16 0.16 0.17" specular="0.04 0.04 0.04"/>
    <map stiffness="1e+04" stiffnessrot="5e+04" force="2e-05" shadowclip="8" shadowscale="1.2" znear="0.002"/>
    <scale jointwidth="0.004" framewidth="0.004"/>
  </visual>

  <statistic meansize="0.05" extent="4" center="0 0 0.3"/>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.32 0.52 0.74" rgb2="0.93 0.90 0.80"
      width="800" height="800"/>
    <texture name="soiltex" type="2d" builtin="flat" rgb1="0.17 0.16 0.11" rgb2="0.11 0.11 0.08"
      width="600" height="600" mark="random" markrgb="0.25 0.28 0.15" random="0.55"/>
    <material name="soil" texture="soiltex" texrepeat="40 40" texuniform="true" specular="0.03" shininess="0.03"/>
    <texture name="stonetex" type="2d" builtin="flat" rgb1="0.31 0.30 0.28" rgb2="0.25 0.24 0.22"
      width="600" height="600" mark="random" markrgb="0.37 0.35 0.32" random="0.34"/>
    <material name="stone" texture="stonetex" texrepeat="5 5" texuniform="true" specular="0.12" shininess="0.15" reflectance="0.02"/>
    <material name="pebble" rgba="0.38 0.36 0.33 1" specular="0.2" shininess="0.3"/>
    <material name="grass4" rgba="0.44 0.64 0.28 1" specular="0.25" shininess="0.35"/>
    <material name="light"    rgba="0.86 0.83 0.74 1" specular="0.35" shininess="0.55" reflectance="0.06"/>
    <material name="dark"     rgba="0.20 0.31 0.26 1" specular="0.35" shininess="0.55" reflectance="0.06"/>
    <material name="rim"      rgba="0.32 0.24 0.16 1" specular="0.3"  shininess="0.5"/>
    <material name="ivory"    rgba="0.88 0.85 0.77 1" specular="0.35" shininess="0.6" reflectance="0.04"/>
    <material name="obsidian" rgba="0.13 0.13 0.15 1" specular="0.55" shininess="0.85" reflectance="0.08"/>
    <material name="grass1"   rgba="0.30 0.52 0.22 1" specular="0.2" shininess="0.3"/>
    <material name="grass2"   rgba="0.38 0.60 0.26 1" specular="0.2" shininess="0.3"/>
    <material name="grass3"   rgba="0.24 0.44 0.20 1" specular="0.2" shininess="0.3"/>
    <material name="dew"      rgba="0.80 0.90 0.95 0.55" specular="0.9" shininess="0.95" reflectance="0.3"/>
  </asset>

  <worldbody>
    <light name="sun" directional="true" pos="-7 -9 7" dir="0.62 0.78 -0.60" diffuse="0.82 0.76 0.63"
      specular="0.30 0.28 0.22" castshadow="true"/>
    <light name="sky" directional="true" pos="2 6 9" dir="-0.2 -0.6 -1" diffuse="0.20 0.25 0.33" castshadow="false"/>
    <light name="bounce" directional="true" pos="6 -2 1" dir="-0.8 0.3 0.25" diffuse="0.16 0.15 0.11" castshadow="false"/>
    <geom name="ground" type="plane" size="30 30 0.2" pos="0 0 -0.30" material="soil"/>
    <geom name="slab" type="cylinder" size="{STONE_R:.3f} 0.15" pos="0 0 -0.15" material="stone"/>
    <geom name="rim_n" type="box" size="{BOARD/2 + RIM:.3f} {RIM/2:.3f} {THICK*1.4:.4f}" pos="0 {BOARD/2 + RIM/2:.3f} {THICK:.4f}" material="rim" contype="0" conaffinity="0" group="1"/>
    <geom name="rim_s" type="box" size="{BOARD/2 + RIM:.3f} {RIM/2:.3f} {THICK*1.4:.4f}" pos="0 {-(BOARD/2 + RIM/2):.3f} {THICK:.4f}" material="rim" contype="0" conaffinity="0" group="1"/>
    <geom name="rim_e" type="box" size="{RIM/2:.3f} {BOARD/2:.3f} {THICK*1.4:.4f}" pos="{BOARD/2 + RIM/2:.3f} 0 {THICK:.4f}" material="rim" contype="0" conaffinity="0" group="1"/>
    <geom name="rim_w" type="box" size="{RIM/2:.3f} {BOARD/2:.3f} {THICK*1.4:.4f}" pos="{-(BOARD/2 + RIM/2):.3f} 0 {THICK:.4f}" material="rim" contype="0" conaffinity="0" group="1"/>
{chr(10).join(squares)}
{chr(10).join(pieces)}
{chr(10).join(grass_blades(rng))}
{chr(10).join(pebbles(rng))}
{chr(10).join(dewdrops(rng))}
  </worldbody>
</mujoco>
"""


def write(path=None, board=None, seed=3):
    path = Path(path or (FLYBODY / "fly_chess_scene.xml"))
    path.write_text(build(board, seed))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scene.png")
    ap.add_argument("--width", type=int, default=1600); ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    import mujoco
    from PIL import Image
    p = write(seed=a.seed)
    m = mujoco.MjModel.from_xml_path(str(p)); d = mujoco.MjData(m)
    # stand the fly on the board's near edge, facing in
    fly = m.joint("free").qposadr[0]
    d.qpos[fly:fly + 7] = [0.0, -(BOARD / 2 + RIM + 0.35), 2 * THICK + 0.132, 0.707, 0, 0, 0.707]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=a.height, width=a.width)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [-0.05, -0.10, 0.14]; cam.distance = 3.9; cam.azimuth = 124; cam.elevation = -12
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(a.out)
    print(f"wrote {a.out}  (board {BOARD:.2f} cm across, square {SQUARE*10:.1f} mm, fly 3.4 mm)")
