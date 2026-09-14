"""The fly's right eye as a screen: lay the chessboard onto the real column map.

MaleCNS assigns every columnar optic-lobe neuron to a column of the eye with two hexagonal
coordinates (annotation columns assignedOlHex1/2). The right eye has 892 columns, one L1 cell
each. Photoreceptors carry no coordinates, so each one is placed in the column of its
strongest postsynaptic partner that does.

The board is the largest axis-aligned square (in the hex lattice's Cartesian embedding) whose
64 cells each contain at least MIN_COLS columns. Columns outside the board form a rim; the
rim is lit when Black is to move.

Each square is painted as light on its columns, in the three receptor channels the eye has:
  R1-R6  luminance:  white piece 1.0, black piece 0.45, empty 0
  R7     ultraviolet / blue channel   } piece kind code (pawn, knight, bishop, rook, queen, king)
  R8     green / blue channel         }   = (1,0) (0,1) (1,1) (.5,0) (0,.5) (.5,.5)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import chess
from .graph import load, DATA

MIN_COLS = 2
AMPLITUDE = 2.0                       # photoreceptor current for a level of 1.0
KIND_CODE = {chess.PAWN: (1.0, 0.0), chess.KNIGHT: (0.0, 1.0), chess.BISHOP: (1.0, 1.0),
             chess.ROOK: (0.5, 0.0), chess.QUEEN: (0.0, 0.5), chess.KING: (0.5, 0.5)}
LUM = {chess.WHITE: 1.0, chess.BLACK: 0.45}
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
        sub = np.abs(W[hexed_idx][:, pr])            # rows: hexed targets, cols: photoreceptors
        sub = sub.tocsc()
        self.pr_col = {}
        for j, p in enumerate(pr):
            col = sub[:, j]
            if col.nnz == 0:
                continue
            best = hexed_idx[col.indices[np.argmax(col.data)]]
            self.pr_col[int(p)] = self.col_index[self.col_of[best]]
        t = meta.type.fillna("")
        self.channel = {}
        for p in self.pr_col:
            ty = t[p]
            if ty == "R1-R6": self.channel[p] = 0
            elif ty.startswith("R7") and not ty.startswith("R7R8"): self.channel[p] = 1
            elif ty.startswith("R8"): self.channel[p] = 2
        self.pr_col = {p: c for p, c in self.pr_col.items() if p in self.channel}
        self.place_board()

    def place_board(self):
        """The board is an 8x8 grid along the hex lattice's own two axes (a parallelogram on
        the eye), placed and sized to cover the most columns with >= MIN_COLS in every cell."""
        h = np.array(self.columns, dtype=float)
        best = None
        for w1 in np.arange(2.0, 5.01, 0.25):
            for w2 in np.arange(2.0, 5.01, 0.25):
                for a in np.arange(h[:, 0].min() - 2, h[:, 0].max() - 8 * w1 + 3, 0.5):
                    for b in np.arange(h[:, 1].min() - 2, h[:, 1].max() - 8 * w2 + 3, 0.5):
                        fx = np.floor((h[:, 0] - a) / w1).astype(int)
                        fy = np.floor((h[:, 1] - b) / w2).astype(int)
                        inside = (fx >= 0) & (fx < 8) & (fy >= 0) & (fy < 8)
                        cells = np.bincount(fy[inside] * 8 + fx[inside], minlength=64)
                        if cells.min() >= MIN_COLS and (best is None or inside.sum() > best[0]):
                            best = (int(inside.sum()), a, b, w1, w2, int(cells.min()), float(cells.mean()))
        covered, a, b, w1, w2, cmin, cmean = best
        fx = np.floor((h[:, 0] - a) / w1).astype(int)
        fy = np.floor((h[:, 1] - b) / w2).astype(int)
        inside = (fx >= 0) & (fx < 8) & (fy >= 0) & (fy < 8)
        self.col_square = np.where(inside, fy * 8 + fx, -1)           # chess.square index: file = fx, rank = fy
        self.board_axial = (float(a), float(b), float(w1), float(w2))
        self.stats = {"columns": len(self.columns), "photoreceptors_placed": len(self.pr_col),
                      "by_channel": {k: int(sum(1 for p in self.channel if self.channel[p] == k)) for k in (0, 1, 2)},
                      "board_columns": covered, "cell_size_hex": [float(w1), float(w2)],
                      "cols_per_square_min": cmin, "cols_per_square_mean": cmean,
                      "rim_columns": int((self.col_square < 0).sum())}

    def encode(self, board: chess.Board):
        """Return the input current vector [N] for one position."""
        levels = np.zeros((len(self.columns), 3), dtype=np.float32)
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc is None:
                continue
            uv, gr = KIND_CODE[pc.piece_type]
            levels[self.col_square == sq] = (LUM[pc.color], uv, gr)
        if board.turn == chess.BLACK:
            levels[self.col_square < 0] = (0.6, 0.6, 0.6)
        I = np.zeros(len(self.meta), dtype=np.float32)
        for p, c in self.pr_col.items():
            I[p] = AMPLITUDE * levels[c, self.channel[p]]
        return I

    def targets(self, board: chess.Board):
        y = np.array([square_state(board.piece_at(sq)) for sq in chess.SQUARES], dtype=np.int64)
        return y, int(board.turn == chess.BLACK)

    def svg(self, path):
        """Draw the column map with the board squares coloured."""
        xy = self.xy; s = 14
        a, b, w1, w2 = self.board_axial
        def cart(h1, h2): return np.array([h1 + 0.5 * h2, h2 * np.sqrt(3) / 2])
        w, h = (xy[:, 0].max() - xy[:, 0].min() + 3) * s, (xy[:, 1].max() - xy[:, 1].min() + 3) * s
        ox, oy = xy[:, 0].min() - 1.5, xy[:, 1].min() - 1.5
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}"><rect width="100%" height="100%" fill="#0A0C11"/>']
        for i, (x, y) in enumerate(xy):
            sq = self.col_square[i]
            if sq < 0:
                fill = "#2A2F3A"
            else:
                f, r = sq % 8, sq // 8
                fill = "#E9E6DD" if (f + r) % 2 else "#8FD3E8"
            out.append(f'<circle cx="{(x - ox) * s:.1f}" cy="{h - (y - oy) * s:.1f}" r="{s * 0.42:.1f}" fill="{fill}"/>')
        def P(h1, h2):
            c = cart(h1, h2); return f'{(c[0] - ox) * s:.1f},{h - (c[1] - oy) * s:.1f}'
        for k in range(9):
            out.append(f'<polyline points="{P(a + k * w1, b)} {P(a + k * w1, b + 8 * w2)}" fill="none" stroke="#F2B841" stroke-width="1"/>')
            out.append(f'<polyline points="{P(a, b + k * w2)} {P(a + 8 * w1, b + k * w2)}" fill="none" stroke="#F2B841" stroke-width="1"/>')
        for f in range(8):
            c = cart(a + (f + 0.5) * w1, b - 1.2); out.append(f'<text x="{(c[0] - ox) * s:.1f}" y="{h - (c[1] - oy) * s:.1f}" fill="#F2B841" font-size="12" text-anchor="middle" font-family="monospace">{"abcdefgh"[f]}</text>')
            c = cart(a - 1.2, b + (f + 0.5) * w2); out.append(f'<text x="{(c[0] - ox) * s:.1f}" y="{h - (c[1] - oy) * s + 4:.1f}" fill="#F2B841" font-size="12" text-anchor="middle" font-family="monospace">{f + 1}</text>')
        out.append(f'<text x="12" y="{h - 10:.0f}" fill="#8B909D" font-size="12" font-family="monospace">right eye · {len(self.columns)} columns · board covers {self.stats["board_columns"]} · grey rim = side-to-move indicator</text></svg>')
        Path(path).write_text("\n".join(out))


if __name__ == "__main__":
    eye = Eye()
    print(json.dumps(eye.stats, indent=1))
    eye.svg(DATA / "eye_board_map.svg")
    b = chess.Board()
    I = eye.encode(b)
    print("start position: driven photoreceptors", int((I > 0).sum()), "of", len(eye.pr_col), "| max current", float(I.max()))
