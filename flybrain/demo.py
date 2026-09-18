"""Phase 6: the fly plays a game in the garden, and you can watch it think.

Plays a real game with the real brain, then animates it. For each move the fly walks from its
resting spot to the piece, carries it to its destination, and walks back to exactly where it
started; a captured piece is carried off the board first. Beside the board, the fly's own brain
lights up with the activity that produced the move, the board is shown as its eye receives it,
and the moves its descending neurons argued for are listed.

Nothing is physically simulated. The fly and the pieces are placed directly each frame. The only
computation is the brain choosing the move.

    python -m flybrain.demo --moves 16 --out data/fly_chess.mp4 --save-game data/demo_game.pkl
    python -m flybrain.demo --replay data/demo_game.pkl --width 960 --height 540 --fps 24   # re-render only
"""
import argparse
import math
import pickle
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
from .panel import Panel, title_card, end_card
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


def render(record, final, out, width, height, fps, device, elo=1320, animate=12):
    board0 = record[0]["board"]
    write(board=board0)                                       # scene XML for the starting position
    m = mujoco.MjModel.from_xml_path(str(FLYBODY / "fly_chess_scene.xml"))
    d = mujoco.MjData(m)
    W, meta = load(); eye = Eye(W, meta)
    anim = Animator(m, d)
    panel = Panel(meta, eye, w=int(width / 3), h=height)
    view_w = width - panel.w
    r = mujoco.Renderer(m, height=height, width=view_w)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)

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
    foot_sub = f"White: the fly · Black: Stockfish {elo}"
    base_look = np.array([-0.05, -0.10, 0.14])

    def frame_3d(az, el, dist, look):
        cam.lookat[:] = look; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        anim.sync(); r.update_scene(d, cam)
        return Image.fromarray(r.render())

    def emit(img):
        nonlocal frames
        wr.append_data(np.asarray(img)); frames += 1

    def shot(state, az, el, dist, look=None, fade=1.0):
        full = Image.new("RGB", (width, height), (11, 13, 19))
        view = frame_3d(az, el, dist, base_look if look is None else look)
        if fade < 1.0:
            view = Image.blend(Image.new("RGB", view.size, (11, 13, 19)), view, fade)
        full.paste(view, (0, 0)); full.paste(panel.render(state), (view_w, 0))
        emit(full)

    # ---- title, then the garden revealed in a slow orbit
    n_title = int(fps * 3.0)
    for i in range(n_title):
        emit(title_card(width, height, i / n_title))
    n_est = int(fps * 4.0)
    for i in range(n_est):
        u = i / max(1, n_est - 1); e = u * u * (3 - 2 * u)
        shot({"mode": "idle", "board": board0, "foot_main": "a chess set built for a fly", "foot_sub": "a 3.2 mm square · a 3.4 mm fly · " + foot_sub},
             az=96 + 28 * e, el=-18 + 9 * e, dist=6.2 - 2.3 * e, fade=min(1.0, i / (fps * 0.8)))

    az_base = 124.0
    n_rest = max(0, len(record) - animate)
    glide_s = min(0.35, max(0.12, 22.0 / max(1, n_rest)))     # the fast-forward: at most ~22 s in all
    for k, step in enumerate(record):
        mv, before = step["move"], step["board"]
        san = before.san(mv)
        drift = 5 * math.sin(k * 0.6)
        who = "the fly" if step["by"] == "fly" else "Stockfish"
        if k >= animate:
            # the rest of the game, quickly: every piece glides, the brain flickers on the fly's moves
            frm, to = square_xy(mv.from_square), square_xy(mv.to_square)
            name = piece_at.get(mv.from_square)
            n = max(2, int(fps * glide_s))
            for i in range(n):
                u = i / (n - 1); e = u * u * (3 - 2 * u)
                if name:
                    anim.place_piece(name, frm[0] + (to[0] - frm[0]) * e, frm[1] + (to[1] - frm[1]) * e, BOARD_TOP + 0.08 * math.sin(math.pi * u))
                shot({"mode": "move" if step["by"] == "fly" else "opponent", "rates": step["snaps"][-1] if step["snaps"] else None, "tick": 24,
                      "board": before, "move": mv, "cands": step["cands"], "san": san, "dim": 0.8,
                      "foot_main": f"move {k + 1} · {who} plays {san}", "foot_sub": "the rest of the game, quickly · " + foot_sub},
                     az=az_base + 30.0 * (k - animate) / max(1, n_rest), el=-12, dist=4.0)
        elif step["by"] == "fly":
            snaps = step["snaps"]; n_s = len(snaps)
            per = max(1, int(fps * THINK / n_s))
            for si, snap in enumerate(snaps):
                for j in range(per):
                    tick = min(24, 3 * (si + 1))
                    shot({"mode": "think", "rates": snap, "tick": tick, "board": before,
                          "cands": step["cands"] if si >= n_s - 2 else None,
                          "foot_main": f"move {k + 1} · the fly is thinking", "foot_sub": foot_sub},
                         az=az_base + drift + 1.5 * (si * per + j) / (n_s * per), el=-11, dist=3.9)
            last = snaps[-1]
            states = segment_frames(walk_segments(mv, before, piece_at), fps)
            n = len(states)
            for i, (x, y, h, ph, carried, cz) in enumerate(states):
                anim.place_fly(x, y, h, ph)
                if carried:
                    anim.place_piece(carried, x - 0.10 * math.sin(h), y + 0.10 * math.cos(h), BOARD_TOP + cz, h)
                ramp = min(1.0, i / (fps * 0.8), (n - 1 - i) / (fps * 0.8))
                tgt = np.array([x, y, BOARD_TOP + 0.10]); look = base_look + (tgt - base_look) * 0.70 * ramp
                shot({"mode": "move", "rates": last, "tick": 24, "board": before, "move": mv, "cands": step["cands"], "san": san, "dim": 0.75,
                      "foot_main": f"move {k + 1} · the fly plays {san}", "foot_sub": foot_sub},
                     az=az_base + drift, el=-11 - 6 * ramp, dist=3.9 - 1.6 * ramp, look=look)
        else:
            # the opponent's piece glides across on its own; a captured piece sinks away as it arrives
            frm, to = square_xy(mv.from_square), square_xy(mv.to_square)
            name = piece_at.get(mv.from_square)
            cap_sq = mv.to_square if before.piece_at(mv.to_square) else (
                (mv.to_square + (-8 if before.turn == chess.WHITE else 8)) if before.is_en_passant(mv) else None)
            victim = piece_at.get(cap_sq) if cap_sq is not None else None
            n = int(fps * 1.1)
            for i in range(n):
                u = i / max(1, n - 1); e = u * u * (3 - 2 * u)
                if name:
                    anim.place_piece(name, frm[0] + (to[0] - frm[0]) * e, frm[1] + (to[1] - frm[1]) * e, BOARD_TOP + 0.10 * math.sin(math.pi * u))
                if victim and u > 0.55:
                    anim.place_piece(victim, DUMP[0] + 0.12 * (k % 5), DUMP[1] - 0.10 * (k % 6), BOARD_TOP)
                shot({"mode": "opponent", "board": before, "move": mv,
                      "foot_main": f"move {k + 1} · Stockfish plays {san}", "foot_sub": foot_sub},
                     az=az_base + drift, el=-11, dist=3.9)
        # commit the move to the scene bookkeeping
        cap_sq = mv.to_square if before.piece_at(mv.to_square) else (
            (mv.to_square + (-8 if before.turn == chess.WHITE else 8)) if before.is_en_passant(mv) else None)
        if cap_sq is not None and cap_sq in piece_at:
            victim = piece_at.pop(cap_sq)
            anim.place_piece(victim, DUMP[0] + 0.12 * (k % 5), DUMP[1] - 0.10 * (k % 6), BOARD_TOP)
        if mv.from_square in piece_at:
            name = piece_at.pop(mv.from_square)
            piece_at[mv.to_square] = name
            x, y = square_xy(mv.to_square); anim.place_piece(name, x, y, BOARD_TOP)
        anim.place_fly(REST[0], REST[1], math.pi / 2, 0.0)
        progress(k + 1, total, t0, "demo: rendering")

    # ---- hold the final position, then the card
    for i in range(int(fps * 2.0)):
        shot({"mode": "idle", "board": final, "foot_main": "the position after " + f"{len(record)} moves", "foot_sub": foot_sub},
             az=az_base + 3, el=-13, dist=4.3)
    res = final.result(claim_draw=True)
    how = ("checkmate" if final.is_checkmate() else "stalemate" if final.is_stalemate() else
           "insufficient material" if final.is_insufficient_material() else "repetition" if final.can_claim_threefold_repetition()
           else "the fifty-move rule" if final.can_claim_fifty_moves() else "")
    verdict = {"1-0": "the fly wins", "0-1": "Stockfish wins", "1/2-1/2": "a draw"}.get(res, f"unfinished after {len(record)} plies")
    if how: verdict += f" · {how}"
    n_moves = sum(1 for x in record if x["by"] == "fly")
    n_end = int(fps * 4.5)
    for i in range(n_end):
        emit(end_card(width, height, i / n_end, verdict, f"{n_moves} moves chosen by the fly · {len(record)} plies played"))
    wr.close()
    print(f"wrote {out}: {frames} frames, {frames/fps:.1f} s, {time.time()-t0:.0f} s to render")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=200, help="plies to play; the game stops earlier when it ends")
    ap.add_argument("--animate", type=int, default=12, help="plies shown in full; the rest are fast-forwarded to the result")
    ap.add_argument("--out", default="data/fly_chess.mp4")
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    ap.add_argument("--stockfish", default="stockfish"); ap.add_argument("--elo", type=int, default=1320)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-game", default=None, help="pickle the played game so the video can be re-rendered without replaying")
    ap.add_argument("--replay", default=None, help="render a previously saved game instead of playing one")
    a = ap.parse_args()
    if a.replay:
        rec, final = pickle.load(open(a.replay, "rb"))
        print(f"replaying {len(rec)} recorded moves from {a.replay}")
    else:
        rec, final = play(a.moves, a.weights, a.stockfish, a.elo, a.device, a.seed)
        print(f"game: {len(rec)} moves recorded, final position {final.fen()}")
        if a.save_game:
            pickle.dump((rec, final), open(a.save_game, "wb")); print(f"saved the game to {a.save_game}")
    render(rec, final, a.out, a.width, a.height, a.fps, a.device, a.elo, a.animate)
