"""The side panel of the demo: the fly's brain lit by its own activity, the board as the fly's
eye receives it, and the moves its descending neurons argued for.

Everything here is drawing. The rates come from the model exactly as Phase 3 built it; this
file only decides where a neuron's dot goes and what colour it is.

    python -m flybrain.panel            # renders a test panel to data/panel_test.png
"""
import base64
import json
import math
import numpy as np
import chess
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from .graph import DATA

BG = (11, 13, 19)
INK = (236, 233, 224)
MUTED = (140, 146, 160)
DIM = (86, 90, 104)
AMBER = (252, 190, 70)
BLUE = (92, 168, 240)
LINE = (38, 42, 54)

# the brain's neuropils, grouped for the ghost map; the nerve cord is left out on purpose
VNC = ("AB(", "ANm", "CV-", "HTct", "IntTct", "LTct", "LegNp", "NTct", "WTct", "mVAC", "Ov(")
GROUPS = [
    ("optic lobe",     ("LA(", "ME(", "LO(", "LOP(", "AME("),                        (30, 84, 104)),
    ("central complex", ("FB", "EB", "PB", "NO"),                                    (104, 66, 132)),
    ("mushroom body",  ("CA(", "PED(", "aL(", "a'L(", "bL(", "b'L(", "gL("),        (128, 60, 92)),
    ("antennal lobe",  ("AL(",),                                                     (92, 96, 46)),
    ("lateral horn",   ("LH(",),                                                     (62, 92, 66)),
    ("gnathal",        ("GNG", "PRW", "SAD", "FLA(", "CAN("),                        (74, 74, 86)),
    ("central brain",  None,                                                         (58, 64, 100)),
]


def _font(name, size):
    paths = {"title": ("/System/Library/Fonts/Supplemental/Georgia.ttf", 0),
             "body": ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
             "bold": ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
             "glyph": ("/System/Library/Fonts/Apple Symbols.ttf", 0)}
    try:
        p, i = paths[name]
        return ImageFont.truetype(p, size, index=i)
    except OSError:
        return ImageFont.load_default()


def group_of(name):
    for g, prefixes, _ in GROUPS:
        if prefixes and any(name.startswith(p) for p in prefixes):
            return g
    return "central brain"


class Panel:
    DESIGN = (640, 1080)                  # everything is laid out at this size and scaled to fit

    def __init__(self, meta, eye, w=640, h=1080):
        self.out_w, self.out_h = w, h
        w, h = self.DESIGN
        self.w, self.h = w, h
        self.f = {k: _font(*v) for k, v in {"title": ("title", 40), "sub": ("body", 15), "h": ("bold", 15), "body": ("body", 14),
                                             "small": ("body", 12), "big": ("bold", 22), "move": ("title", 30), "glyph": ("glyph", 22),
                                             "label": ("body", 12)}.items()}
        # ---- layout, top to bottom
        self.y_head = 0; self.y_brain = 92; self.brain_h = int(h * 0.40)
        self.y_eye = self.y_brain + self.brain_h + 22; self.eye_h = int(h * 0.245)
        self.y_cand = self.y_eye + self.eye_h + 18; self.cand_h = int(h * 0.135)
        self.y_foot = h - 78
        self._build_brain(meta)
        self._build_eye(eye)

    # ------------------------------------------------------------------ the brain map
    def _build_brain(self, meta):
        cloud = json.load(open(DATA / "cloud.json"))
        roi = np.frombuffer(base64.b64decode(cloud["roi"]["pos"]), np.int16).reshape(-1, 3).astype(np.float32) * 0.1
        lab = np.frombuffer(base64.b64decode(cloud["roi"]["lab"]), np.int16)
        names = {r["id"]: r["name"] for r in cloud["roi"]["regions"]}
        brain_ids = [i for i, n in names.items() if not n.startswith(VNC)]
        keep = np.isin(lab, brain_ids)
        roi, lab = roi[keep], lab[keep]
        soma = np.frombuffer(base64.b64decode(cloud["soma"]["pos"]), np.int16).reshape(-1, 3).astype(np.float32) * 0.1
        self.soma_index = meta.soma_index.to_numpy()
        zmax = roi[:, 2].max() + 30.0
        has = (self.soma_index >= 0)
        own = soma[np.where(has, self.soma_index, 0)]
        has &= own[:, 2] <= zmax                                      # somas in the head only
        self.has = has
        self.idx = self.soma_index[has]
        # frontal projection: x across, y down (the atlas y already runs dorsal to ventral)
        pts = np.concatenate([roi[:, :2], own[has][:, :2]], 0)
        lo, hi = pts.min(0), pts.max(0)
        pad_x, pad_top, pad_bot = 26, 40, 34
        sc = min((self.w - 2 * pad_x) / (hi[0] - lo[0]), (self.brain_h - pad_top - pad_bot) / (hi[1] - lo[1]))
        span = (hi - lo) * sc
        off = np.array([pad_x + ((self.w - 2 * pad_x) - span[0]) / 2, pad_top + ((self.brain_h - pad_top - pad_bot) - span[1]) / 2])
        to_px = lambda p: ((p[:, :2] - lo) * sc + off)
        # ghost: a density image per group, tinted
        ghost = np.zeros((self.brain_h, self.w, 3), np.float32)
        gnames = np.array([group_of(names[int(l)]) for l in np.unique(lab)]); gid = {int(l): g for l, g in zip(np.unique(lab), gnames)}
        gcol = {g: np.array(c, np.float32) for g, _, c in GROUPS}
        px = to_px(roi).astype(np.int32)
        px[:, 0] = np.clip(px[:, 0], 0, self.w - 1); px[:, 1] = np.clip(px[:, 1], 0, self.brain_h - 1)
        glab = np.array([gid[int(l)] for l in lab])
        self.centroids, self.extent = {}, {}
        for g, _, c in GROUPS:
            m = glab == g
            if not m.any(): continue
            dens = np.zeros((self.brain_h, self.w), np.float32)
            np.add.at(dens, (px[m, 1], px[m, 0]), 1.0)
            dens = gaussian_filter(dens, 1.6)
            a = np.clip(dens / (np.percentile(dens[dens > 0], 90) + 1e-6), 0, 1) ** 0.6
            ghost += a[:, :, None] * gcol[g][None, None, :]
            self.centroids[g] = px[m].mean(0); self.extent[g] = (px[m].min(0), px[m].max(0))
        # the two optic lobes separately, and which of them is the eye the board is shown to
        side = np.array([names[int(l)][-2] if names[int(l)].endswith(")") else "" for l in lab])
        ol = glab == "optic lobe"
        self.lobes = {s: px[ol & (side == s)].mean(0) for s in ("L", "R") if (ol & (side == s)).any()}
        ghost = np.clip(ghost * 0.55, 0, 255)
        base = np.zeros((self.brain_h, self.w, 3), np.float32); base[:, :] = BG
        self.ghost = np.clip(base + ghost, 0, 255).astype(np.uint8)
        sp = to_px(own[has]).astype(np.int32)
        self.px = np.zeros((len(soma), 2), np.int32)
        self.px[self.idx] = np.stack([np.clip(sp[:, 0], 1, self.w - 3), np.clip(sp[:, 1], 1, self.brain_h - 3)], 1)
        # the descending neurons: the readout's source, lit on their own scale so they show
        sc_ = meta.superclass.to_numpy()
        self.dn = np.flatnonzero((sc_ == "descending_neuron") & has)
        self.dn_px = self.px[self.soma_index[self.dn]]
        # which lobe sees the board: the one holding the right eye's photoreceptors
        pr = np.flatnonzero((sc_ == "ol_sensory") & (meta.side.to_numpy() == "R") & has)
        prx = self.px[self.soma_index[pr], 0].mean() if len(pr) else self.w / 2
        self.seeing = min(self.lobes, key=lambda k: abs(self.lobes[k][0] - prx)) if self.lobes else None

    def draw_brain(self, img, rates, tick, ticks=24, dim=1.0, live_label=True):
        arr = self.ghost.astype(np.float32)
        n_active = 0
        if rates is not None:
            v = rates[self.has]
            peak = float(np.abs(v).max()) or 1.0
            live = np.flatnonzero(np.abs(v) > 0.004 * peak)
            n_active = len(live)
            glow = np.zeros_like(arr)
            if len(live):
                a = np.clip(np.abs(v[live]) / peak, 0, 1) ** 0.35 * dim
                warm = v[live] > 0
                col = np.where(warm[:, None], np.array(AMBER, np.float32), np.array(BLUE, np.float32)) * a[:, None]
                x, y = self.px[self.idx[live], 0], self.px[self.idx[live], 1]
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    np.maximum.at(glow, (y + dy, x + dx), col)
            # the descending neurons, on their own scale: they are the readout
            dv = rates[self.dn]; dpeak = float(np.abs(dv).max()) or 1.0
            da = np.clip(np.abs(dv) / dpeak, 0, 1) ** 0.5 * dim
            dcol = np.where((dv > 0)[:, None], np.array((255, 236, 200), np.float32), np.array((170, 210, 255), np.float32)) * da[:, None]
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):
                np.maximum.at(glow, (np.clip(self.dn_px[:, 1] + dy, 0, self.brain_h - 1), np.clip(self.dn_px[:, 0] + dx, 0, self.w - 1)), dcol)
            halo = gaussian_filter(glow, (2.2, 2.2, 0)) * 1.6
            arr = np.maximum(arr, np.clip(arr + halo, 0, 255)); arr = np.maximum(arr, glow)
        img.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), (0, self.y_brain))
        dr = ImageDraw.Draw(img)
        y0 = self.y_brain
        dr.text((22, y0 + 8), "THE BRAIN", font=self.f["h"], fill=INK)
        dr.text((22, y0 + 28), "144,209 neurons · 21 million connections · every one measured", font=self.f["small"], fill=MUTED)
        # region labels
        lf = self.f["label"]; lc = (176, 182, 196)
        for s_, (cx, cy) in self.lobes.items():
            lab = "right optic lobe · sees the board" if s_ == self.seeing else ("left optic lobe" if s_ == "L" else "right optic lobe")
            if s_ != self.seeing: lab = {"L": "left", "R": "right"}[s_] + " optic lobe"
            tw = dr.textlength(lab, font=lf)
            (lo_, hi_) = self.extent["optic lobe"]
            dr.text((min(max(8, cx - tw / 2), self.w - tw - 8), y0 + hi_[1] + 6), lab, font=lf, fill=lc)
        if "mushroom body" in self.extent:
            (lo_, hi_), (cx, cy) = self.extent["mushroom body"], self.centroids["mushroom body"]
            tw = dr.textlength("mushroom bodies", font=lf)
            dr.text((cx - tw / 2, y0 + lo_[1] - 16), "mushroom bodies", font=lf, fill=lc)
        if "central complex" in self.extent:
            (lo_, hi_), (cx, cy) = self.extent["central complex"], self.centroids["central complex"]
            tw = dr.textlength("central complex", font=lf)
            dr.text((cx - tw / 2, y0 + hi_[1] + 4), "central complex", font=lf, fill=lc)
        if len(self.dn_px):
            cx, cy = self.dn_px.mean(0); tw = dr.textlength("descending neurons", font=lf)
            dr.text((cx - tw / 2, y0 + cy + 30), "descending neurons", font=lf, fill=lc)
        # activity readout under the map
        yb = y0 + self.brain_h - 26
        dr.line([(22, yb - 8), (self.w - 22, yb - 8)], fill=LINE)
        if rates is not None and live_label:
            dr.text((22, yb), f"tick {tick} of {ticks}", font=self.f["body"], fill=INK)
            frac = n_active / max(1, int(self.has.sum()))
            bx0, bx1 = 130, self.w - 190
            dr.rectangle([bx0, yb + 6, bx1, yb + 12], fill=LINE)
            dr.rectangle([bx0, yb + 6, bx0 + int((bx1 - bx0) * min(1.0, tick / ticks)), yb + 12], fill=(120, 126, 140))
            dr.text((self.w - 176, yb), f"{n_active:,} neurons active", font=self.f["small"], fill=MUTED)
        else:
            dr.text((22, yb), "at rest", font=self.f["body"], fill=DIM)
        dr.text((self.w - 22 - dr.textlength("amber above rest · blue below", font=self.f["small"]), y0 + 30),
                "amber above rest · blue below", font=self.f["small"], fill=DIM)

    # ------------------------------------------------------------------ the eye and the board
    def _build_eye(self, eye):
        xy = eye.xy.copy(); self.col_square = eye.col_square
        half = (self.w - 3 * 22) / 2
        self.box_w = half; self.box_h = self.eye_h - 46
        # the hex lattice comes out as a sheared strip; turn it so its long axis lies across the box
        ctr = xy.mean(0); u, sv, vt = np.linalg.svd(xy - ctr, full_matrices=False)
        rot = vt                                               # rows: principal axes
        if rot[0, 0] < 0: rot[0] = -rot[0]
        if np.cross(np.r_[rot[0], 0], np.r_[rot[1], 0])[2] < 0: rot[1] = -rot[1]
        q = (xy - ctr) @ rot.T
        lo, hi = q.min(0), q.max(0)
        sc = min((half - 16) / (hi[0] - lo[0]), (self.box_h - 16) / (hi[1] - lo[1]))
        self.hex_r = sc * 0.56
        c = ((q - lo) * sc) + 8
        c[:, 0] += (half - 16 - (hi[0] - lo[0]) * sc) / 2; c[:, 1] += (self.box_h - 16 - (hi[1] - lo[1]) * sc) / 2
        c[:, 1] = self.box_h - c[:, 1]                         # dorsal up
        self.hex_c = c
        theta = math.atan2(rot[0, 1], rot[0, 0])
        ang = np.linspace(0, 2 * math.pi, 7)[:-1] + math.pi / 6 - theta
        self.hex_off = np.stack([np.cos(ang), -np.sin(ang)], 1) * self.hex_r
        from .eye import STATE_CODE
        self.state_code = STATE_CODE

    def draw_eye(self, img, board, move=None, cands=None):
        dr = ImageDraw.Draw(img)
        y0 = self.y_eye; x_b = 22; x_e = 22 + self.box_w + 22
        dr.text((x_b, y0), "THE BOARD", font=self.f["h"], fill=INK)
        dr.text((x_e, y0), "ON THE RETINA", font=self.f["h"], fill=INK)
        dr.text((x_b, y0 + 20), "White: the fly", font=self.f["small"], fill=MUTED)
        dr.text((x_e, y0 + 20), "892 columns of the right eye, 645 show the board", font=self.f["small"], fill=MUTED)
        top = y0 + 42
        # the board diagram
        sq = min(self.box_w, self.box_h) / 8
        bx = x_b + (self.box_w - 8 * sq) / 2; by = top + (self.box_h - 8 * sq) / 2
        frm, to = (move.from_square, move.to_square) if move else (None, None)
        for s in chess.SQUARES:
            f, r = chess.square_file(s), chess.square_rank(s)
            x, y = bx + f * sq, by + (7 - r) * sq
            light = (f + r) % 2 == 1
            col = (214, 200, 170) if light else (122, 88, 60)
            if s in (frm, to): col = (240, 190, 90) if light else (190, 140, 60)
            dr.rectangle([x, y, x + sq, y + sq], fill=col)
        glyph = {chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝", chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚"}
        gf = self.f["glyph"]
        for s, pc in board.piece_map().items():
            f, r = chess.square_file(s), chess.square_rank(s)
            x, y = bx + f * sq + sq / 2, by + (7 - r) * sq + sq / 2
            g = glyph[pc.piece_type]
            tw = dr.textlength(g, font=gf)
            if pc.color == chess.WHITE:
                dr.text((x - tw / 2, y - 13), g, font=gf, fill=(250, 248, 240), stroke_width=1, stroke_fill=(60, 40, 20))
            else:
                dr.text((x - tw / 2, y - 13), g, font=gf, fill=(20, 18, 20), stroke_width=1, stroke_fill=(180, 170, 150))
        if move:
            (fx, fy), (tx, ty) = [(bx + chess.square_file(s) * sq + sq / 2, by + (7 - chess.square_rank(s)) * sq + sq / 2) for s in (frm, to)]
            dr.line([(fx, fy), (tx, ty)], fill=AMBER, width=4)
            ang = math.atan2(ty - fy, tx - fx)
            dr.polygon([(tx, ty), (tx - 12 * math.cos(ang - 0.45), ty - 12 * math.sin(ang - 0.45)),
                        (tx - 12 * math.cos(ang + 0.45), ty - 12 * math.sin(ang + 0.45))], fill=AMBER)
        # the retina: each column coloured by the two colour photoreceptors' levels
        lvl = np.zeros((len(self.hex_c), 2), np.float32)
        pm = board.piece_map()
        for i, s in enumerate(self.col_square):
            if s >= 0 and s in pm:
                pc = pm[s]; lvl[i] = self.state_code[(pc.color, pc.piece_type)]
        for i, (cx, cy) in enumerate(self.hex_c):
            s = self.col_square[i]
            if s < 0:
                col = (22, 25, 33)
            else:
                r7, r8 = lvl[i]
                base = (48, 52, 64) if (chess.square_file(s) + chess.square_rank(s)) % 2 == 1 else (36, 39, 50)
                col = (int(base[0] + r7 * 200), int(base[1] + r7 * 130 + r8 * 60), int(base[2] + r8 * 190))
                if s in (frm, to): col = tuple(min(255, c + 40) for c in col)
            pts = [(x_e + cx + ox, top + cy + oy) for ox, oy in self.hex_off]
            dr.polygon(pts, fill=col)
        dr.text((x_e, top + self.box_h + 4), "amber: white pieces (R7)  ·  blue: black pieces (R8)", font=self.f["small"], fill=DIM)

    # ------------------------------------------------------------------ the candidates
    def draw_candidates(self, img, cands, chosen=None):
        dr = ImageDraw.Draw(img)
        y = self.y_cand
        dr.line([(22, y - 6), (self.w - 22, y - 6)], fill=LINE)
        dr.text((22, y + 4), "WHAT THE DESCENDING NEURONS ARGUED FOR", font=self.f["h"], fill=INK)
        dr.text((22, y + 24), "1,314 neurons carry every command from the brain to the body; the move is read from them", font=self.f["small"], fill=MUTED)
        if not cands:
            return
        y += 50
        top = max(1e-6, cands[0][1])
        for san, p in cands[:3]:
            bar = int((self.w - 150) * min(1.0, p / top))
            col = AMBER if san == chosen else (70, 92, 112)
            dr.rectangle([22, y + 4, 22 + bar, y + 18], fill=col)
            dr.text((30, y + 2), san, font=self.f["body"], fill=(20, 20, 24) if san == chosen and bar > 60 else INK)
            dr.text((self.w - 96, y + 2), f"{100 * p:5.1f}%", font=self.f["body"], fill=INK if san == chosen else MUTED)
            y += 26

    # ------------------------------------------------------------------ header and footer
    def draw_frame(self, img, header, sub, foot_main, foot_sub):
        dr = ImageDraw.Draw(img)
        dr.text((22, 14), "FLY CHESS", font=self.f["title"], fill=INK)
        tw = dr.textlength("FLY CHESS", font=self.f["title"])
        dr.text((22, 64), sub, font=self.f["sub"], fill=MUTED)
        dr.line([(22, self.y_foot - 10), (self.w - 22, self.y_foot - 10)], fill=LINE)
        dr.text((22, self.y_foot), foot_main, font=self.f["move"], fill=INK)
        dr.text((22, self.y_foot + 42), foot_sub, font=self.f["body"], fill=MUTED)

    def render(self, state):
        """state: mode, rates, tick, board, move, cands, header, move_no, who, san, caption."""
        img = Image.new("RGB", (self.w, self.h), BG)
        mode = state.get("mode", "idle")
        self.draw_frame(img, "FLY CHESS", "a fruit fly's connectome plays chess · every neuron and connection as measured",
                        state.get("foot_main", ""), state.get("foot_sub", ""))
        self.draw_brain(img, state.get("rates"), state.get("tick", 0), dim=state.get("dim", 1.0),
                        live_label=mode in ("think", "move"))
        self.draw_eye(img, state["board"], state.get("move") if mode in ("move", "opponent") else None)
        self.draw_candidates(img, state.get("cands"), chosen=state.get("san") if mode == "move" else None)
        if (self.out_w, self.out_h) != (self.w, self.h):
            img = img.resize((self.out_w, self.out_h), Image.LANCZOS)
        return img


# ---------------------------------------------------------------------- full-frame cards
def title_card(w, h, t, fonts=None):
    """The opening card; t in [0, 1] fades it in and out."""
    img = Image.new("RGB", (w, h), BG); dr = ImageDraw.Draw(img)
    a = min(1.0, t / 0.18, (1 - t) / 0.18) if t < 1 else 0
    def mix(c): return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))
    f1, f2, f3 = _font("title", 96), _font("body", 26), _font("body", 18)
    lines = [("FLY CHESS", f1, mix(INK), -120), ("a fruit fly's complete brain map, made to play", f2, mix(MUTED), 10),
             ("144,209 neurons · 21 million connections · every one exactly as measured · only their strengths learned", f3, mix(DIM), 62),
             ("White: the fly    Black: Stockfish", f3, mix(DIM), 96)]
    for txt, f, c, dy in lines:
        tw = dr.textlength(txt, font=f)
        dr.text(((w - tw) / 2, h / 2 + dy - (60 if f is f1 else 0)), txt, font=f, fill=c)
    return img


def end_card(w, h, t, result, moves, fonts=None):
    img = Image.new("RGB", (w, h), BG); dr = ImageDraw.Draw(img)
    a = min(1.0, t / 0.25)
    def mix(c): return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))
    f1, f2, f3 = _font("title", 64), _font("body", 24), _font("body", 17)
    lines = [(result, f1, mix(INK), -70), (moves, f2, mix(MUTED), 20),
             ("nothing in the demo is simulated but the brain: the fly sees the board, 144,209 neurons run for 24 ticks,", f3, mix(DIM), 80),
             ("and the move is read from its 1,314 descending neurons. The walk is animation.", f3, mix(DIM), 106),
             ("MaleCNS v1.0 · Janelia, Cambridge, Google Research · github.com/HakimElAyoubi/fly-chess", f3, mix(DIM), 160)]
    for txt, f, c, dy in lines:
        tw = dr.textlength(txt, font=f)
        dr.text(((w - tw) / 2, h / 2 + dy - 40), txt, font=f, fill=c)
    return img


if __name__ == "__main__":
    from .graph import load
    from .eye import Eye
    W, meta = load(); eye = Eye(W, meta)
    p = Panel(meta, eye, w=640, h=1080)
    S = "/private/tmp/claude-501/-Users-hakim-Desktop-fruit-fly/87f30919-7987-456d-9e6d-4dae2ee87d86/scratchpad/snap.npz"
    z = np.load(S, allow_pickle=True)
    snaps, fen, cands = z["snaps"], str(z["fen"]), [(str(a), float(b)) for a, b in z["cands"]]
    b = chess.Board(fen); mv = b.parse_san(cands[0][0])
    think = p.render({"mode": "think", "rates": snaps[3], "tick": 12, "board": b, "cands": None,
                      "foot_main": "move 2 · the fly is thinking", "foot_sub": "White: the fly · Black: Stockfish 1320"})
    move = p.render({"mode": "move", "rates": snaps[-1], "tick": 24, "board": b, "move": mv, "cands": cands, "san": cands[0][0], "dim": 0.7,
                     "foot_main": f"move 2 · the fly plays {cands[0][0]}", "foot_sub": "White: the fly · Black: Stockfish 1320"})
    out = Image.new("RGB", (640 * 2 + 20, 1080), (40, 40, 40)); out.paste(think, (0, 0)); out.paste(move, (660, 0))
    out.save(DATA / "panel_test.png"); print("wrote data/panel_test.png")
