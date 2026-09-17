"""The digital fly as a UCI chess engine (Phase 4).

    python -m flybrain.uci [--weights data/train_gpu/model_step11600.pt] [--temperature 0]

Speaks the UCI protocol on stdin/stdout so any chess GUI or match runner can play it. One
forward pass of the fly's brain per move (about two seconds on a laptop CPU, a few
milliseconds on a GPU); the move head is masked to legal moves and the promotion head picks
the piece when a pawn reaches the last rank. The fly sees only the pieces and whose turn it
is, so castling rights and en passant come from the legal-move mask, not from its eye.
"""
import argparse
import sys
import numpy as np
import torch
import chess
from .graph import load, DATA
from .eye import Eye
from .policy import FlyPolicy, legal_mask, load_into, variant_of, N_MOVES, PROMO_INV

NAME = "Fly Chess (MaleCNS digital fly)"


class FlyEngine:
    def __init__(self, weights, temperature=0.0, device="cpu", avoid_repetition=True):
        torch.set_grad_enabled(False)
        edge, ck = variant_of(weights, device)
        self.model = FlyPolicy(device=device, ticks=24, edge_gains=edge)
        load_into(self.model, ck["model"], weights, strict_report=False)
        self.model.eval()
        self.step = ck.get("step")
        W, meta = load(); self.eye = Eye(W, meta)
        self.temperature = temperature
        self.device = device
        self.avoid_repetition = avoid_repetition            # a policy without search repeats itself; refuse moves that recreate a position

    def choose(self, boards):
        """Batched move choice for a list of boards (all with legal moves). Returns chess.Move list."""
        I = torch.from_numpy(np.stack([self.eye.encode(b) for b in boards], 1)).to(self.device)
        logits, plog, vlog, _ = self.model(I)
        logits = logits + legal_mask(boards, self.device)
        if self.avoid_repetition:
            for k, b in enumerate(boards):
                repeats, others = [], []
                for m in b.legal_moves:
                    b.push(m); (repeats if b.is_repetition(2) else others).append(m); b.pop()
                if repeats and others:
                    for m in repeats:
                        logits[k, m.from_square * 64 + m.to_square] = float("-inf")
        if self.temperature > 0:
            idx = torch.multinomial(torch.softmax(logits / self.temperature, 1), 1).squeeze(1)
        else:
            idx = logits.argmax(1)
        promo = plog[:, 1:].argmax(1) + 1                        # queen, rook, bishop, knight
        moves = []
        for b, i, p in zip(boards, idx.tolist(), promo.tolist()):
            m = chess.Move(i // 64, i % 64)
            if m not in b.legal_moves:                           # a pawn reaching the last rank needs a promotion piece
                m = chess.Move(i // 64, i % 64, promotion=PROMO_INV[p])
                if m not in b.legal_moves:
                    m = chess.Move(i // 64, i % 64, promotion=chess.QUEEN)
            moves.append(m)
        return moves, vlog.softmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--allow-repetition", action="store_true")
    a = ap.parse_args()
    engine = FlyEngine(a.weights, a.temperature, a.device, avoid_repetition=not a.allow_repetition)
    board = chess.Board()
    out = lambda s: (sys.stdout.write(s + "\n"), sys.stdout.flush())
    for line in sys.stdin:
        cmd = line.strip().split()
        if not cmd:
            continue
        if cmd[0] == "uci":
            out(f"id name {NAME} step {engine.step}"); out("id author Hakim El Ayoubi")
            out("option name Temperature type spin default 0 min 0 max 200"); out("uciok")
        elif cmd[0] == "isready":
            out("readyok")
        elif cmd[0] == "setoption" and len(cmd) >= 5 and cmd[2].lower() == "temperature":
            engine.temperature = float(cmd[4]) / 100.0
        elif cmd[0] == "ucinewgame":
            board = chess.Board()
        elif cmd[0] == "position":
            if cmd[1] == "startpos":
                board = chess.Board(); rest = cmd[2:]
            else:
                fen = " ".join(cmd[2:8]); board = chess.Board(fen); rest = cmd[8:]
            if rest and rest[0] == "moves":
                for u in rest[1:]:
                    board.push_uci(u)
        elif cmd[0] == "go":
            if board.is_game_over():
                out("bestmove 0000")
            else:
                (m,), v = engine.choose([board])
                out(f"info string value win {v[0, 2]:.2f} draw {v[0, 1]:.2f} loss {v[0, 0]:.2f}")
                out(f"bestmove {m.uci()}")
        elif cmd[0] == "quit":
            break


if __name__ == "__main__":
    main()
