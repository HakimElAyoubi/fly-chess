"""A batched chess environment for reinforcement learning (Phase 7).

Everything so far taught the fly by imitation: it was shown a position and told which move a
human, and later Stockfish, had played. That never tells it whether the move was any good for
*it*, only whether it matched. Reinforcement learning closes that loop: the fly plays its own
games, and the only teaching signal is how those games turn out.

One environment step is a full ply pair: the fly moves, the opponent replies, and the fly is
handed the position it now has to deal with. Many games run at once so that every decision
point in the batch goes through the brain in a single forward pass, which is what makes this
affordable at 144,209 neurons a position.

Reward
  terminal      +1 win, 0 draw, -1 loss, from the fly's point of view
  shaping       potential-based (Ng, Harada & Russell 1999): gamma * phi(s') - phi(s) with
                phi = tanh(material advantage / 5), and phi = 0 at a terminal state. Shaping of
                this form cannot change which policy is optimal, it only tells the fly sooner
                that it is winning material, which matters here because its games are long and
                its wins are rare.

Finished games are immediately replaced by fresh ones, so the batch never shrinks and the
forward pass is always full width.
"""
import chess
import numpy as np

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATERIAL_SCALE = 5.0            # a rook up is worth about tanh(1) = 0.76 of the potential


def material(board, colour):
    """Material advantage of `colour`, in pawns."""
    s = 0
    for sq, pc in board.piece_map().items():
        s += PIECE_VALUE[pc.piece_type] * (1 if pc.color == colour else -1)
    return s


def potential(board, colour):
    return float(np.tanh(material(board, colour) / MATERIAL_SCALE))


def greedy_move(board, rng):
    """Mate in one if there is one, else the most valuable capture, else a random legal move."""
    best, best_val = [], -1
    for m in board.legal_moves:
        board.push(m); mate = board.is_checkmate(); board.pop()
        if mate:
            return m
        v = 0
        if board.is_capture(m):
            v = 1 if board.is_en_passant(m) else PIECE_VALUE[board.piece_at(m.to_square).piece_type]
        if v > best_val:
            best, best_val = [m], v
        elif v == best_val:
            best.append(m)
    return best[int(rng.integers(len(best)))]


def outcome(board, fly_colour):
    """Terminal reward from the fly's point of view, or None if the game is still on."""
    if not board.is_game_over(claim_draw=True):
        return None
    if board.is_checkmate():
        return 1.0 if board.turn != fly_colour else -1.0
    return 0.0


class BatchChessEnv:
    """`n` games in parallel. observe() returns the positions waiting on the fly; step() plays the
    fly's moves, lets the opponent reply, and returns one reward per position."""

    def __init__(self, n=32, opponent="greedy", opening_plies=4, max_plies=200, seed=0,
                 gamma=0.99, shaping=0.5, stockfish="stockfish", sf_depth=1, sf_elo=1320):
        self.n, self.opponent, self.opening_plies, self.max_plies = n, opponent, opening_plies, max_plies
        self.gamma, self.shaping = gamma, shaping
        self.rng = np.random.default_rng(seed)
        self.engine = None
        if opponent == "stockfish":
            import chess.engine
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish)
            self.engine.configure({"Threads": 1} if sf_depth else {"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
            self.sf_limit = chess.engine.Limit(depth=sf_depth) if sf_depth else chess.engine.Limit(time=0.02)
        self.games = [self._new(i) for i in range(n)]
        self.finished = []                       # (result, plies) of every game that ended, for the caller

    # ---------------------------------------------------------------- game lifecycle
    def _new(self, i):
        """A fresh game: random opening plies for variety, colours alternating, opponent moves first
        if the fly is black, so that observe() always hands back a position the fly must play."""
        b = chess.Board()
        for _ in range(self.opening_plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[int(self.rng.integers(len(moves)))])
        g = {"board": b, "colour": chess.WHITE if i % 2 == 0 else chess.BLACK, "plies": 0}
        if b.turn != g["colour"]:
            self._opponent_move(g)
        if outcome(b, g["colour"]) is not None or b.is_game_over(claim_draw=True):
            return self._new(i + 1)              # the random opening ended the game outright: try again
        g["phi"] = potential(b, g["colour"])
        return g

    def _opponent_move(self, g):
        b = g["board"]
        if b.is_game_over(claim_draw=True):
            return
        if self.opponent == "greedy":
            b.push(greedy_move(b, self.rng))
        elif self.opponent == "random":
            moves = list(b.legal_moves); b.push(moves[int(self.rng.integers(len(moves)))])
        else:
            b.push(self.engine.play(b, self.sf_limit).move)

    # ---------------------------------------------------------------- the RL interface
    def observe(self):
        """The `n` positions the fly has to move in, one per slot."""
        return [g["board"] for g in self.games]

    def colours(self):
        return [g["colour"] for g in self.games]

    def step(self, moves):
        """Play one move per slot, let the opponent reply, and score the transition.
        Returns (reward, done) arrays of length n. A done slot is refilled with a new game, so the
        board returned by the next observe() is the start of a new episode."""
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        for i, (g, m) in enumerate(zip(self.games, moves)):
            b = g["board"]
            b.push(m); g["plies"] += 1
            r = outcome(b, g["colour"])
            if r is None and g["plies"] < self.max_plies:
                self._opponent_move(g); g["plies"] += 1
                r = outcome(b, g["colour"])
                if g["plies"] >= self.max_plies and r is None:
                    r = 0.0                                          # too long: call it a draw
            elif r is None:
                r = 0.0
            if r is None:                                            # still going: shaping only
                phi = potential(b, g["colour"])
                rew[i] = self.shaping * (self.gamma * phi - g["phi"])
                g["phi"] = phi
            else:                                                    # terminal: phi(terminal) = 0
                rew[i] = r + self.shaping * (0.0 - g["phi"])
                done[i] = True
                self.finished.append(((r + 1) / 2, g["plies"], "white" if g["colour"] else "black"))
                self.games[i] = self._new(int(self.rng.integers(1 << 30)))
        return rew, done

    def close(self):
        if self.engine is not None:
            self.engine.quit()
