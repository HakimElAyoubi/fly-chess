"""Phase 6: the fly plays a game in the garden, and you can watch it think.

Plays a real game with the real brain, then animates it. For each move the fly walks from its
resting spot to the piece, carries it to its destination, and walks back to exactly where it
started; a captured piece is carried off the board first. Beside the board, the fly's own brain
lights up with the activity that produced the move.

Nothing is physically simulated. The fly and the pieces are placed directly each frame. The only
computation is the brain choosing the move.

    python -m flybrain.demo --moves 12 --out data/fly_chess.mp4
    python -m flybrain.demo --moves 6 --width 960 --height 600 --fps 24   # quicker draft
"""
import argparse
import base64
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import chess
import chess.engine
import mujoco
from PIL import Image, ImageDraw, ImageFont
from .graph import load, DATA
from .eye import Eye
from .policy import FlyPolicy, variant_of, load_into, legal_mask, PROMO_INV
from .scene import build, write, SQUARE, BOARD, THICK, RIM, square_xy, FLYBODY
from .progress import write as progress

BOARD_TOP = 2 * THICK
STAND = 0.1235                  # thorax height above the feet in the fly's standing pose
REST = (-(BOARD / 2 + RIM + 0.30), -(BOARD / 2 + RIM + 0.22))    # where the fly waits between moves
DUMP = (BOARD / 2 + RIM + 0.30, BOARD / 2 * 0.2)                 # where captured pieces are set down
SPEED = 1.25                    # cm per second, a deliberate fly walk
THINK = 1.9                     # seconds of thinking before each move
GRIP = 0.45                     # seconds to take hold of a piece and to let go
LEGS = [("T1", "left"), ("T2", "right"), ("T3", "left"), ("T1", "right"), ("T2", "left"), ("T3", "right")]


# ----------------------------------------------------------------------------- the fly's walk
def quat_from_heading(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


class Animator:
    """Places the fly and the pieces. Never steps physics."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.fly_q = model.joint("free").qposadr[0]
        self.piece_q = {}
        for i in range(model.njnt):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n and n.endswith("_free") and n != "free":
                self.piece_q[n[:-5]] = model.joint(n).qposadr[0]
        self.leg = {}
        for seg, joint in (("coxa", "coxa"), ("femur", "femur"), ("tibia", "tibia")):
            for t, s in LEGS:
                nm = f"{joint}_{t}_{s}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
                if jid >= 0:
                    self.leg[(seg, t, s)] = model.jnt_qposadr[jid]

    def place_fly(self, x, y, heading, gait_phase, airborne=0.0):
        q = self.fly_q
        self.d.qpos[q:q + 3] = [x, y, BOARD_TOP + STAND + airborne]
        self.d.qpos[q + 3:q + 7] = quat_from_heading(heading)
        for i, (t, s) in enumerate(LEGS):
            ph = gait_phase + (0.0 if i < 3 else math.pi)
            swing = max(0.0, math.sin(ph))
            for seg, amp, bias in (("coxa", 0.34, 0.0), ("femur", 0.42, 0.0), ("tibia", -0.26, 0.0)):
                k = (seg, t, s)
                if k not in self.leg:
                    continue
                v = bias + (amp * swing if seg == "femur" else amp * math.sin(ph))
                self.d.qpos[self.leg[k]] = v

    def place_piece(self, name, x, y, z, heading=0.0):
        q = self.piece_q.get(name)
        if q is None:
            return
        self.d.qpos[q:q + 3] = [x, y, z]
        self.d.qpos[q + 3:q + 7] = quat_from_heading(heading)

    def hide(self, name):
        self.place_piece(name, 60.0, 60.0, -20.0)

    def sync(self):
        mujoco.mj_forward(self.m, self.d)


# ----------------------------------------------------------------------------- move choreography
def walk_segments(move, board_before, piece_at):
    """A list of (kind, start_xy, end_xy, carried_piece) describing the whole trip."""
    frm, to = square_xy(move.from_square), square_xy(move.to_square)
    captured = board_before.piece_at(move.to_square)
    ep = board_before.is_en_passant(move)
    segs = []
    here = REST
    if captured is not None or ep:
        victim_sq = move.to_square if not ep else (move.to_square + (-8 if board_before.turn == chess.WHITE else 8))
        victim = piece_at.get(victim_sq)
        vx, vy = square_xy(victim_sq)
        segs.append(("walk", here, (vx, vy), None))
        segs.append(("grip", (vx, vy), (vx, vy), None))
        segs.append(("walk", (vx, vy), DUMP, victim))
        segs.append(("drop", DUMP, DUMP, victim))
        here = DUMP
    mover = piece_at.get(move.from_square)
    segs.append(("walk", here, frm, None))
    segs.append(("grip", frm, frm, None))
    segs.append(("walk", frm, to, mover))
    segs.append(("drop", to, to, mover))
    segs.append(("walk", to, REST, None))
    return segs


def segment_frames(segs, fps):
    """Expand segments into per-frame (x, y, heading, phase, carried, carry_z) states."""
    out = []
    phase = 0.0
    heading = math.pi / 2
    for kind, a, b, carried in segs:
        if kind == "walk":
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(2, int(round(fps * dist / SPEED)))
            h = math.atan2(b[1] - a[1], b[0] - a[0]) - math.pi / 2
            for i in range(n):
                u = i / (n - 1)
                s = u * u * (3 - 2 * u)                      # ease in and out
                x = a[0] + (b[0] - a[0]) * s; y = a[1] + (b[1] - a[1]) * s
                phase += 2 * math.pi * (dist / n) / 0.075     # one stride per 0.75 mm
                hh = heading + (h - heading) * min(1.0, u * 3) if i < n // 3 else h
                out.append((x, y, hh, phase, carried, 0.055))
            heading = h
        else:
            n = max(2, int(round(fps * GRIP)))
            for i in range(n):
                u = i / (n - 1)
                lift = math.sin(u * math.pi / 2) if kind == "grip" else math.cos(u * math.pi / 2)
                out.append((a[0], a[1], heading, phase, carried, 0.055 * lift if carried else 0.055))
    return out


# ----------------------------------------------------------------------------- the brain panel
class BrainPanel:
    """The brain itself, lit by its rates, with the fly's actual move candidates underneath."""

    def __init__(self, meta, w=380, h=470):
        cloud = json.load(open(DATA / "cloud.json"))
        pos = np.frombuffer(base64.b64decode(cloud["soma"]["pos"]), np.int16).astype(np.float32).reshape(-1, 3) * 0.1
        self.w, self.h = w, h
        self.brain_h = int(h * 0.52)
        self.soma = meta.soma_index.to_numpy()
        self.has = self.soma >= 0
        self.idx = self.soma[self.has]
        # scale to the neurons this model actually contains (the brain), not the whole nervous system
        own = pos[self.idx]
        xy = np.stack([own[:, 0], -own[:, 1]], 1)
        lo, hi = xy.min(0), xy.max(0)
        pad_x, pad_top, pad_bot = 18, 62, 16
        sc = min((w - 2 * pad_x) / (hi[0] - lo[0]), (self.brain_h - pad_top - pad_bot) / (hi[1] - lo[1]))
        span = (hi - lo) * sc
        off = np.array([pad_x + ((w - 2 * pad_x) - span[0]) / 2,
                        pad_top + ((self.brain_h - pad_top - pad_bot) - span[1]) / 2])
        self.px = np.zeros((len(pos), 2), np.int32)
        self.px[self.idx] = ((xy - lo) * sc + off).astype(np.int32)
        self.px[:, 0] = np.clip(self.px[:, 0], 1, w - 3); self.px[:, 1] = np.clip(self.px[:, 1], 1, self.brain_h - 3)
        # a dim ghost of every neuron, so the lit parts are seen inside the whole brain
        self.ghost = np.zeros((h, w, 3), np.uint8); self.ghost[:, :] = (10, 12, 17)
        gx, gy = self.px[self.idx, 0], self.px[self.idx, 1]
        np.maximum.at(self.ghost, (gy, gx), np.array([23, 26, 34], np.uint8))
        np.maximum.at(self.ghost, (gy + 1, gx), np.array([17, 19, 26], np.uint8))
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
            self.mid = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
            self.small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
        except OSError:
            self.font = self.mid = self.small = ImageFont.load_default()

    def draw(self, rates, caption="", dim=1.0, candidates=None, header=""):
        arr = self.ghost.copy()
        if rates is not None:
            v = rates[self.has]
            peak = float(np.abs(v).max()) or 1.0
            live = np.flatnonzero(np.abs(v) > 0.015 * peak)
            if len(live):
                a = np.clip(np.abs(v[live]) / peak, 0, 1) ** 0.5 * dim
                warm = v[live] > 0
                col = np.where(warm[:, None], np.array([252, 196, 78]), np.array([96, 172, 236]))
                col = (col * a[:, None]).astype(np.uint8)
                x, y = self.px[self.idx[live], 0], self.px[self.idx[live], 1]
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    np.maximum.at(arr, (y + dy, x + dx), col)
        img = Image.fromarray(arr)
        dr = ImageDraw.Draw(img)
        dr.text((18, 14), "the fly's brain", font=self.font, fill=(233, 230, 221))
        dr.text((18, 36), "144,209 neurons · the board is shown to the right eye", font=self.small, fill=(120, 126, 140))
        y = self.brain_h + 4
        dr.line([(18, y), (self.w - 18, y)], fill=(40, 44, 54)); y += 16
        dr.text((18, y), header or "", font=self.mid, fill=(233, 230, 221)); y += 26
        dr.text((18, y), caption, font=self.small, fill=(150, 154, 165)); y += 24
        if candidates:
            dr.text((18, y), "what the descending neurons wanted", font=self.small, fill=(120, 126, 140)); y += 20
            for san, p in candidates:
                bar = int((self.w - 120) * min(1.0, p / max(1e-6, candidates[0][1])))
                dr.rectangle([18, y + 3, 18 + bar, y + 12], fill=(60, 80, 96))
                dr.text((18, y), f"{san}", font=self.mid, fill=(233, 230, 221))
                dr.text((self.w - 78, y), f"{100*p:5.1f}%", font=self.small, fill=(150, 154, 165))
                y += 22
        dr.text((18, self.h - 22), "amber above rest · blue below", font=self.small, fill=(90, 94, 105))
        return img


# ----------------------------------------------------------------------------- the game
def play(n_moves, weights, stockfish, elo, device, seed):
    edge, ck = variant_of(weights, device)
    model = FlyPolicy(device=device, ticks=24, edge_gains=edge)
    load_into(model, ck["model"], weights, strict_report=False); model.eval()
    W, meta = load(); eye = Eye(W, meta)
    sf = chess.engine.SimpleEngine.popen_uci(stockfish)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1})
    board = chess.Board()
    rng = np.random.default_rng(seed)
    for _ in range(int(rng.integers(0, 3)) * 2):             # a little opening variety
        board.push(list(board.legal_moves)[int(rng.integers(len(list(board.legal_moves))))])
    record = []
    t0 = time.time()
    for k in range(n_moves):
        if board.is_game_over():
            break
        if board.turn == chess.WHITE:                        # the fly plays White
            I = torch.from_numpy(np.stack([eye.encode(board)], 1)).to(device)
            snaps = model.run_trace(I, every=3)
            with torch.no_grad():
                logits, plog, _, _ = model(I)
            logits = logits + legal_mask([board], device)
            for m in board.legal_moves:                      # do not shuffle a piece back and forth
                board.push(m); rep = board.is_repetition(2); board.pop()
                if rep and len(list(board.legal_moves)) > 1:
                    logits[0, m.from_square * 64 + m.to_square] = float("-inf")
            probs = torch.softmax(logits[0], 0)
            top = torch.topk(probs, 4)
            cands = []
            for pr, ix in zip(top.values.tolist(), top.indices.tolist()):
                cm = chess.Move(ix // 64, ix % 64)
                if cm not in board.legal_moves:
                    cm = chess.Move(ix // 64, ix % 64, promotion=chess.QUEEN)
                if cm in board.legal_moves:
                    cands.append((board.san(cm), pr))
            i = int(logits.argmax(1))
            mv = chess.Move(i // 64, i % 64)
            if mv not in board.legal_moves:
                mv = chess.Move(i // 64, i % 64, promotion=PROMO_INV[int(plog[0, 1:].argmax()) + 1])
                if mv not in board.legal_moves:
                    mv = chess.Move(i // 64, i % 64, promotion=chess.QUEEN)
            record.append({"by": "fly", "move": mv, "board": board.copy(), "snaps": snaps, "cands": cands[:3]})
        else:
            mv = sf.play(board, chess.engine.Limit(time=0.05)).move
            record.append({"by": "stockfish", "move": mv, "board": board.copy(), "snaps": None, "cands": None})
        board.push(mv)
        progress(k + 1, n_moves, t0, "demo: playing")
        print(f"  {k+1:3d}. {record[-1]['by']:9s} {mv.uci()}", flush=True)
    sf.quit()
    return record, board


def render(record, out, width, height, fps, device):
    board0 = record[0]["board"]
    write(board=board0)                                       # scene XML for the starting position
    m = mujoco.MjModel.from_xml_path(str(FLYBODY / "fly_chess_scene.xml"))
    d = mujoco.MjData(m)
    W, meta = load()
    anim = Animator(m, d)
    panel = BrainPanel(meta, w=int(width * 0.30), h=height)
    view_w = width - panel.w
    r = mujoco.Renderer(m, height=height, width=view_w)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)

    # which scene body currently holds which square
    piece_at = {sq: f"p{sq}" for sq in board0.piece_map()}
    spares = [f"spare{i}" for i in range(4)]
    for sq, name in piece_at.items():
        x, y = square_xy(sq); anim.place_piece(name, x, y, BOARD_TOP)
    for s in spares:
        anim.hide(s)
    anim.place_fly(REST[0], REST[1], math.pi / 2, 0.0)
    anim.sync()

    import imageio.v2 as imageio
    wr = imageio.get_writer(out, fps=fps, quality=8, macro_block_size=None)
    t0 = time.time(); total = len(record); frames = 0

    def shot(caption, rates, focus=None, close=0.0, dim=1.0, cands=None, header=""):
        nonlocal frames
        base = np.array([-0.05, -0.10, 0.14])
        if focus is not None:
            tgt = np.array([focus[0], focus[1], BOARD_TOP + 0.10])
            cam.lookat[:] = base + (tgt - base) * close
        else:
            cam.lookat[:] = base
        cam.distance = 3.9 - 1.5 * close
        anim.sync()
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        full = Image.new("RGB", (width, height), (10, 12, 17))
        full.paste(img, (0, 0)); full.paste(panel.draw(rates, caption, dim, cands, header), (view_w, 0))
        wr.append_data(np.asarray(full)); frames += 1

    for k, step in enumerate(record):
        mv, before = step["move"], step["board"]
        cam.azimuth = 124 + 6 * math.sin(k * 0.5); cam.elevation = -12
        san = before.san(mv)
        if step["by"] == "fly":
            for s, snap in enumerate(step["snaps"]):
                for _ in range(max(1, int(fps * THINK / len(step["snaps"])))):
                    shot(f"thinking · tick {3*(s+1)} of 24", snap,
                         cands=step["cands"] if s >= len(step["snaps"]) - 2 else None,
                         header=f"move {k+1} · the fly")
            last = step["snaps"][-1]
            states = segment_frames(walk_segments(mv, before, piece_at), fps)
            n = len(states)
            for i, (x, y, h, ph, carried, cz) in enumerate(states):
                anim.place_fly(x, y, h, ph)
                if carried:
                    anim.place_piece(carried, x - 0.10 * math.sin(h), y + 0.10 * math.cos(h), BOARD_TOP + cz, h)
                ramp = min(1.0, i / (fps * 0.8), (n - 1 - i) / (fps * 0.8))     # ease the camera in and out
                doing = "carrying the piece" if carried else "walking"
                shot(doing, last, focus=(x, y), close=0.70 * ramp, dim=0.7,
                     cands=step["cands"], header=f"move {k+1} · the fly plays {san}")
        else:
            for _ in range(int(fps * 0.9)):
                shot("the opponent replies", None, header=f"move {k+1} · Stockfish plays {san}")
        # commit the move to the scene bookkeeping
        cap_sq = mv.to_square if before.piece_at(mv.to_square) else (
            (mv.to_square + (-8 if before.turn == chess.WHITE else 8)) if before.is_en_passant(mv) else None)
        if cap_sq is not None and cap_sq in piece_at:
            victim = piece_at.pop(cap_sq)
            anim.place_piece(victim, DUMP[0] + 0.12 * (len(spares) % 5), DUMP[1] - 0.10 * (k % 6), BOARD_TOP)
        if mv.from_square in piece_at:
            name = piece_at.pop(mv.from_square)
            piece_at[mv.to_square] = name
            x, y = square_xy(mv.to_square); anim.place_piece(name, x, y, BOARD_TOP)
        anim.place_fly(REST[0], REST[1], math.pi / 2, 0.0)
        progress(k + 1, total, t0, "demo: rendering")
    for _ in range(fps):
        shot("", None, header="")
    wr.close()
    print(f"wrote {out}: {frames} frames, {frames/fps:.1f} s, {time.time()-t0:.0f} s to render")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=12)
    ap.add_argument("--out", default="data/fly_chess.mp4")
    ap.add_argument("--width", type=int, default=1440); ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    ap.add_argument("--stockfish", default="stockfish"); ap.add_argument("--elo", type=int, default=1320)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rec, final = play(a.moves, a.weights, a.stockfish, a.elo, a.device, a.seed)
    print(f"game: {len(rec)} moves recorded, final position {final.fen()}")
    render(rec, a.out, a.width, a.height, a.fps, a.device)
