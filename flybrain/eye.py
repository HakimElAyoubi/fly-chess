"""The fly's right eye as a screen: lay the chessboard onto the real column map.

MaleCNS assigns every columnar optic-lobe neuron to a column of the eye with two hexagonal
coordinates (annotation columns assignedOlHex1/2). The right eye has 892 columns, one L1 cell
each. Photoreceptors carry no coordinates, so each one is placed in the column of its
strongest postsynaptic partner that does.

The board is a retinotopic quantile grid over the columns that have both colour receptors
traced: 8 rank bands of equal column count by dorsoventral position, each split into 8 files
by anteroposterior position.

Each square is painted as light on its columns, in the three receptor channels the eye has.
The two colour channels carry the whole 13-way state as a point on a 4x4 grid of levels
(0, 1/3, 2/3, 1): empty (0,0); white pieces run along the R7 edge and black pieces along the
R8 edge. R1-R6 luminance carries the side to move as a global brightness bias (0.6 everywhere
when Black is to move), because only 380 of the 892 columns have traced R1-R6 cells while R7
and R8 cover about 650-700.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import chess
from .graph import load, DATA

MIN_COLS = 2
MIN_PR = 2                            # R7 and R8 photoreceptors in every square
RATE_MAX = 0.8                        # photoreceptor rate (tanh) at level 1.0; currents are atanh(RATE_MAX * level)
LINEARIZE = True                      # so the four levels give equally spaced rates 0, .27, .53, .8, not 0, .58, .87, .96
def current(level):
    return float(np.arctanh(RATE_MAX * level)) if LINEARIZE else 2.0 * level
_L = (0.0, 1 / 3, 2 / 3, 1.0)
# (R7, R8) levels for each of the 12 piece states; empty is (0, 0)
STATE_CODE = {(chess.WHITE, chess.PAWN): (_L[1], 0), (chess.WHITE, chess.KNIGHT): (_L[2], 0), (chess.WHITE, chess.BISHOP): (1, 0),
              (chess.WHITE, chess.ROOK): (1, _L[1]), (chess.WHITE, chess.QUEEN): (1, _L[2]), (chess.WHITE, chess.KING): (1, 1),
              (chess.BLACK, chess.PAWN): (0, _L[1]), (chess.BLACK, chess.KNIGHT): (0, _L[2]), (chess.BLACK, chess.BISHOP): (0, 1),
              (chess.BLACK, chess.ROOK): (_L[1], 1), (chess.BLACK, chess.QUEEN): (_L[2], 1), (chess.BLACK, chess.KING): (_L[2], _L[2])}
TURN_LUM = 0.6                        # R1-R6 level everywhere when Black is to move
# strictly one-per-column cell types on the right eye: photoreceptors are placed by their
# strongest partner among these, so a wide-field partner cannot pull many receptors into one column
COLUMNAR_TYPES = {"C2", "C3", "L1", "L2", "L3", "L5", "Mi1", "Mi4", "Mi9", "T1", "Tm1", "Tm2", "Tm20", "Tm4", "Tm9"}
# 13 square states: 0 empty, 1-6 white P N B R Q K, 7-12 black P N B R Q K
def square_state(piece):
    if piece is None:
        return 0
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


class Eye:
    def __init__(self, W=None, meta=None, side="R"):
        if W is None:
            W, meta = load()
        self.meta = meta
        ann = pd.read_feather(DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather")[["bodyId", "assignedOlHex1", "assignedOlHex2"]]
        m = meta.merge(ann, on="bodyId", how="left")
        hexed = m[m.assignedOlHex1.notna() & (m.side == side)]
        self.col_of = dict(zip(hexed.idx, zip(hexed.assignedOlHex1.astype(int), hexed.assignedOlHex2.astype(int))))
        cols = sorted(set(self.col_of.values()))
        self.columns = cols
        self.col_index = {c: i for i, c in enumerate(cols)}
        # Cartesian embedding of the axial hex coordinates
        h = np.array(cols, dtype=float)
        self.xy = np.stack([h[:, 0] + 0.5 * h[:, 1], h[:, 1] * np.sqrt(3) / 2], 1)
        # photoreceptors -> column of the strongest hexed postsynaptic partner
        pr = meta.index[(meta.superclass == "ol_sensory") & (meta.side == side)].to_numpy()
        hexed_idx = np.array(sorted(self.col_of))
        t = meta.type.fillna("")
        columnar = np.array([t[i] in COLUMNAR_TYPES for i in hexed_idx])
        sub = np.abs(W[hexed_idx][:, pr])            # rows: hexed targets, cols: photoreceptors
        sub = sub.tocsc()
        self.pr_col = {}
        for j, p in enumerate(pr):
            col = sub[:, j]
            if col.nnz == 0:
                continue
            ok = columnar[col.indices]
            idx, dat = (col.indices[ok], col.data[ok]) if ok.any() else (col.indices, col.data)
            best = hexed_idx[idx[np.argmax(dat)]]
            self.pr_col[int(p)] = self.col_index[self.col_of[best]]
        self.channel = {}
        for p in self.pr_col:
            ty = t[p]
            if ty == "R1-R6": self.channel[p] = 0
            elif ty.startswith("R7") and not ty.startswith("R7R8"): self.channel[p] = 1
            elif ty.startswith("R8"): self.channel[p] = 2
        self.pr_col = {p: c for p, c in self.pr_col.items() if p in self.channel}
        self.place_board()

    def place_board(self):
        """Retinotopic quantile mapping: the eye's columns are split by their dorsoventral
        position into 8 rank bands of equal count, and each band by anteroposterior position
        into 8 files of equal count. Every square gets about 892/64 = 14 columns and the whole
        eye is the board (no rim); the side to move is a global brightness bias instead."""
        xy = self.xy
        pr_cols = np.array([self.pr_col[p] for p in sorted(self.pr_col)])
        pr_chan = np.array([self.channel[p] for p in sorted(self.pr_col)])
        has = np.zeros((len(xy), 3), bool); has[pr_cols, pr_chan] = True
        seeing = np.where(has[:, 1] & has[:, 2])[0]                     # columns with both colour receptors
        n = len(seeing)
        order_y = seeing[np.argsort(xy[seeing, 1], kind="stable")]
        rank = np.full(len(xy), -1, int); rank[order_y] = np.minimum(np.arange(n) * 8 // n, 7)
        file = np.full(len(xy), -1, int)
        for r in range(8):
            members = np.where(rank == r)[0]
            order_x = members[np.argsort(xy[members, 0], kind="stable")]
            file[order_x] = np.minimum(np.arange(len(members)) * 8 // len(members), 7)
        self.col_square = np.where(rank >= 0, rank * 8 + file, -1)      # chess.square index: file + 8 * rank; -1 = unused column
        cov = np.zeros((64, 3), int)
        on = self.col_square[pr_cols] >= 0
        np.add.at(cov, (self.col_square[pr_cols][on], pr_chan[on]), 1)
        cells = np.bincount(self.col_square[self.col_square >= 0], minlength=64)
        self.coverage = cov
        if cov[:, 1:].min() < MIN_PR:
            grid = lambda k: "\n".join(" ".join(f"{cov[r * 8 + f, k]:2d}" for f in range(8)) for r in range(7, -1, -1))
            raise RuntimeError("some square has fewer than %d R7 or R8 photoreceptors\nR7 per square (rank 8 at top):\n%s\nR8:\n%s" % (MIN_PR, grid(1), grid(2)))
        self.stats = {"columns": len(self.columns), "photoreceptors_placed": len(self.pr_col),
                      "by_channel": {k: int(sum(1 for p in self.channel if self.channel[p] == k)) for k in (0, 1, 2)},
                      "board_columns": int(n), "unused_columns": int(len(xy) - n), "cols_per_square_min": int(cells.min()), "cols_per_square_mean": float(cells.mean()),
                      "photoreceptors_per_square_min_by_channel": cov.min(0).tolist(),
                      "photoreceptors_per_square_mean_by_channel": cov.mean(0).round(1).tolist(),
                      "squares_without_R1-6": int((cov[:, 0] == 0).sum())}

    def encode(self, board: chess.Board):
        """Return the input current vector [N] for one position."""
        levels = np.zeros((len(self.columns), 3), dtype=np.float32)
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc is None:
                continue
            uv, gr = STATE_CODE[(pc.color, pc.piece_type)]
            levels[self.col_square == sq, 1:] = (uv, gr)
        if board.turn == chess.BLACK:
            levels[:, 0] = TURN_LUM                                       # global brightness bias on R1-R6
        I = np.zeros(len(self.meta), dtype=np.float32)
        for p, c in self.pr_col.items():
            I[p] = current(levels[c, self.channel[p]])
        return I

    def targets(self, board: chess.Board):
        y = np.array([square_state(board.piece_at(sq)) for sq in chess.SQUARES], dtype=np.int64)
        return y, int(board.turn == chess.BLACK)

    def svg(self, path):
        """Draw the column map with the board squares coloured and labelled."""
        xy = self.xy; s = 14
        w, h = (xy[:, 0].max() - xy[:, 0].min() + 3) * s, (xy[:, 1].max() - xy[:, 1].min() + 3) * s
        ox, oy = xy[:, 0].min() - 1.5, xy[:, 1].min() - 1.5
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}"><rect width="100%" height="100%" fill="#0A0C11"/>']
        for i, (x, y) in enumerate(xy):
            f, r = self.col_square[i] % 8, self.col_square[i] // 8
            fill = "#2A2F3A" if self.col_square[i] < 0 else "#E9E6DD" if (f + r) % 2 else "#8FD3E8"
            out.append(f'<circle cx="{(x - ox) * s:.1f}" cy="{h - (y - oy) * s:.1f}" r="{s * 0.42:.1f}" fill="{fill}"/>')
        for sq in range(64):
            c = xy[self.col_square == sq].mean(0)
            out.append(f'<text x="{(c[0] - ox) * s:.1f}" y="{h - (c[1] - oy) * s + 4:.1f}" fill="#B4562E" font-size="11" text-anchor="middle" font-family="monospace" font-weight="bold">{chess.square_name(sq)}</text>')
        out.append(f'<text x="12" y="{h - 10:.0f}" fill="#8B909D" font-size="12" font-family="monospace">right eye · {len(self.columns)} columns · board on the {self.stats["board_columns"]} with both colour receptors · {self.stats["cols_per_square_mean"]:.1f} columns per square · grey = no colour receptor traced</text></svg>')
        Path(path).write_text("\n".join(out))


if __name__ == "__main__":
    eye = Eye()
    print(json.dumps(eye.stats, indent=1))
    eye.svg(DATA / "eye_board_map.svg")
    b = chess.Board()
    I = eye.encode(b)
    print("start position: driven photoreceptors", int((I > 0).sum()), "of", len(eye.pr_col), "| max current", float(I.max()))
