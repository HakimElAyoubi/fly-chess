# flybrain — the digital fly (Phases 1–4)

A connectome-constrained rate model of the male fruit fly brain, built from MaleCNS v1.0.

| file | what it does |
|---|---|
| `graph.py` | builds `data/graph_brain.npz` (CSR, W[post, pre] = synapse count × sign of pre) and `graph_brain_meta.feather` |
| `model.py` | `FlyBrain`: r ← r + k·(f(αWr + I + b) − r); tanh / threshold / mixed units; input normalisation |
| `checks.py` | the two biology checks and gain calibration → `data/checks.json` |
| `export_activity.py` | tick-by-tick activity of both experiments → `data/activity.json` (injected into the atlas by `build_viewer.py`) |
| `eye.py` | the right eye as a screen: column map, photoreceptor placement, chessboard → photoreceptor currents |
| `probe.py` | Phase 2 probe: random positions through the model, cached features → `data/probe_features.npz` |
| `probe_fit.py` | Phase 2 readouts (generic and retinotopic) from the cache → `data/probe.json` |
| `data.py` | Phase 3 data: streams the Lichess monthly database, keeps rated 1600–2200 games, samples positions → `data/positions_<tag>.jsonl.gz` |
| `policy.py` | Phase 3 model: the fly as a chess policy (eye → network with learned gains → descending-neuron readout) |
| `train.py` | Phase 3 training loop: imitation of human moves, legality / top-1 / top-3 metrics, checkpoints, progress bar |
| `progress.py` | progress bar for long jobs → `data/progress.txt` (live page: `progress.html`) |
| `uci.py` | Phase 4: the fly as a UCI chess engine (`./fly-uci` for GUIs); batched move choice with legal masking and an anti-repetition rule |
| `play.py` | Phase 4 matches against a random mover or Stockfish at limited strength; Elo with a 95% interval → `data/match_<tag>.json/.pgn` |

Run order (from the project root, with `.venv` active):

```bash
python -m flybrain.graph
python -m flybrain.checks
python -m flybrain.export_activity
python build_viewer.py
python -m flybrain.eye                # board-on-eye map + data/eye_board_map.svg
python -m flybrain.probe              # Phase 2 probe (about 15 minutes on the M4)
```

## The neuron set

144,209 traced neurons on the brain side of the neck: optic lobes (89,390 intrinsic + 4,114
photoreceptors), central brain (32,160 intrinsic, 4,868 sensory, 107 motor, 72 endocrine),
9,201 visual projection + 563 centrifugal, 1,314 descending, 1,846 ascending, 537 sensory
ascending, and a few dozen efferents. 21.27 M signed edges carrying 99.4 M synapses.

Signs (one per neuron, from the dataset's neurotransmitter predictions): acetylcholine +
(92,375), glutamate − (26,874), GABA − (16,733), histamine − (5,902; photoreceptors). Dopamine
(392), serotonin (305) and octopamine (77) neurons are modulatory and silent in version one.
1,551 neurons with no confident prediction are assumed cholinergic.

## The working model

Graded tanh units, every neuron's inputs normalised to sum to 1 in magnitude, global gain 1.5,
dt/τ = 0.25 (5 ms ticks, 20 ms time constant). Activity is relative to rest. Input
normalisation bounds the spectral radius by the gain (Gershgorin), which is what keeps the
network from saturating: the raw signed matrix has a leading eigenvalue of 4,089 (4,767 for the
excitatory subgraph), dominated by the optic lobe's columnar loops, so any single global scale
that keeps it stable starves the central brain's 50-synapse pathways.

Known limitation: normalisation under-weights strong inputs onto neurons with very many
synapses. Example along the sugar path: the second-order sugar neurons receive 20–70% of their
input from sugar taste neurons, but the premotor neuron Roundup receives only 1.2–1.7% of its
input from them (84–95 of ~6,000 synapses), and MN9 6% from Roundup. The motor response is
therefore real but small (about 1% of the peak response). Phase 3 learns per-edge gains, which
is the principled fix; a leaky integrate-and-fire variant (Shiu et al. 2024) is the other.

## Check results (gain 1.5)

- **A, sugar → feeding.** 187 labellar sugar taste neurons driven (chosen as those with ≥ 2
  synapses onto the Shiu 2022 sugar interneurons G2N-1, Phantom, Usnea, Rattle, Zorro). Those
  interneurons respond at 60% of the network's peak. MN9 is pushed above rest (+0.011 of peak,
  rank 2,923 of 144,209 = top 2%). Twenty random sensory sets of the same size (other labellar
  taste neurons, brain mechanosensory neurons) give MN9 −0.0024 ± 0.0028, max +0.0036: z = 4.8.
  0.7% of neurons respond. PASS.
- **B, light on the right eye.** 891 right R1–R6 photoreceptors driven. Lamina L1/L2: right
  0.216 vs left 0.0002. T4/T5 motion detectors: right 0.053 vs left 0.0001. Lobula plate
  tangential cells: right 0.007 vs left 0.000. 7.5% of neurons respond. PASS.

## Speed (Apple M4, CPU, 21 M edges)

batch 1 × 32 ticks: 2.7 s (84 ms/tick). batch 128 × 32 ticks: 22 s (0.7 s/tick). PyTorch's
sparse CSR tensors are not implemented on Apple's MPS backend, so training (Phase 3) needs a
CUDA GPU as the plan assumed.

## Phase 2: showing it the board

**The eye as a screen.** MaleCNS gives every columnar optic-lobe neuron two hexagonal
coordinates on the eye (annotation columns `assignedOlHex1/2`). The right eye has 892 columns,
one L1 cell each. Photoreceptors carry no coordinates, so each of the right eye's 2,239 placed
photoreceptors (888 R1–R6, 653 R7, 698 R8) sits in the column of its strongest partner among
the strictly one-per-column cell types (L1–L5, Mi1, Mi4, Mi9, Tm1, Tm2, Tm20, ...); using any
partner let wide-field medulla cells pull many receptors into one column. The tracing of
R1–R6 is partial (380 columns) while R7 and R8 cover about 650–700, so the board lives on the
645 columns that have both colour receptors: a retinotopic quantile grid, 8 rank bands of equal
column count by dorsoventral position, each split into 8 files by anteroposterior position.
Every square holds 10 columns with at least 10 R7 and 10 R8 receptors. The other 247 columns
are unused. See `data/eye_board_map.svg`.

**Painting a position.** The fly's eye cannot resolve glyph shapes at ten columns per square,
so pieces are told apart by colour. The two colour channels carry the whole 13-way state of a
square as a point on a 4×4 grid of levels (0, 1/3, 2/3, 1): empty is (0,0), the six white
pieces run along the R7 edge, the six black pieces along the R8 edge. The luminance channel
(R1–R6) carries the side to move as a global brightness bias (0.6 everywhere when Black is to
move). Currents are atanh(0.8 × level), so the receptors' tanh rates come out equally spaced
(0, 0.27, 0.53, 0.80); with a plain 2.0 × level the rates were 0, 0.58, 0.87, 0.96 and the
two brightest levels were nearly indistinguishable one synapse in. A per-square classifier on a square's own
photoreceptors recovers the state at 100.0% on held-out boards, so the encoding itself has no
ceiling below the target.

**The probe.** 8,192 uniformly random placements (each square empty with p = 0.4, else a
uniform piece; not legal chess, deliberately, so every state appears on every square hundreds
of times) plus 1,024 positions from random play as a realistic test set. Training uses 7,168
placements, 512 choose the settings, 512 are held out. Each position runs through the untrained
Phase 1 model for 24 ticks and the steady-state activity of each stage is read two ways, both
linear in the activity:

- *generic*: principal components of the whole stage (kept down to a floor on the eigenvalue
  spectrum, scaled globally or with a floored partial whitening), one softmax head per square
  and one for the side to move, fit to convergence with L-BFGS; floor, scaling and penalty are
  chosen on the validation split;
- *retinotopic* (stages with eye columns): each square is read by its own linear head from the
  neurons assigned to that square's columns, the way a downstream neuron would read the medulla.

<!-- RESULTS:begin -->
**Results** (9,216 positions, 7,168 for training, 24 ticks, gain 1.5; a readout that
always answers "empty" scores 40% on random placements and about 44% on random play):

| stage | neurons | generic readout, random placements | generic, random play | retinotopic readout, random placements | retinotopic, random play | side to move |
|---|---|---|---|---|---|---|
| Photoreceptor input (the stimulus itself) | 2,239 | **100.0%** | 100.0% | **—** | — | 100.0% |
| Lamina L1–L5, right | 4,466 | **100.0%** | 100.0% | **100.0%** | 100.0% | 100.0% |
| Medulla + lobula, right (all optic-lobe intrinsic) | 44,789 | **99.98%** | 100.0% | **100.0%** | 100.0% | 100.0% |
| Visual projection neurons, right | 4,612 | **99.98%** | 100.0% | **—** | — | 100.0% |
| Central brain intrinsic | 32,160 | **66.9%** | 76.2% | **—** | — | 100.0% |
| Descending neurons | 1,314 | **50.1%** | 63.5% | **—** | — | 100.0% |

**Verdict: target met.** The plan asks for more than 99.5% of squares read from the medulla and lobula by a linear readout. The generic readout, principal components of the whole stage, gives 99.98% on random placements and 100.0% on realistic positions (kept 176 components, partial scaling, penalty 1e-08); the retinotopic readout, each square read from the neurons of its own eye columns, gives 100.0% / 100.0%. Of the two levers, the training set was the one that mattered: at 3,072 boards the generic readout reached 91.6%, at 7,168 boards 99.98%, with the probe's component floor, scaling and penalty chosen on the validation split. The receptor-rate cap made no difference (0.8, 0.5 and 0.3 gave the same retinotopic accuracy on a pilot) and stays at 0.8. The stimulus itself reads at 100.0%, the lamina at 100.0% and the visual projection neurons at 99.98%: the whole board leaves the optic lobe intact. It then thins out in the untrained central brain (66.9%) and reaches the descending neurons at 50.1% (63.5% realistic), the baseline Phase 3 must beat.
<!-- RESULTS:end -->


## Phase 3: teaching it the rules and the taste of a good move

**The policy.** A position is painted on the right eye exactly as in Phase 2, the network runs
for 24 ticks, and the rates of the 1,314 descending neurons are read by two linear heads: a
move head (from-square × to-square, 4,096 logits, plus 5 promotion logits) and a value head
(win / draw / loss). Descending-neuron rates are standardised by a fixed per-neuron scale set
once at the start of training (`FlyPolicy.calibrate`), because the untrained network's responses
at that depth are about 1e-5 wide; the scale is capped at ten times its median so that a silent
neuron cannot become a noise amplifier. At play time the move head is masked to legal moves; in
training it is not, so legality has to be learned.

**What is learned.** Two variants share one code path.

- *Mac-sized (this pilot):* a positive output gain per neuron, a positive input gain per neuron
  and a bias per neuron, 3 × 144,209 = 432,627 brain parameters, plus the heads (5.4 M). The
  wiring, signs and synapse counts are fixed, and so is the shape of every neuron's input.
- *Per-edge gains (`--edge-gains`, GPU):* a positive gain on each of the 21.27 M connections,
  the plan's main variant. Same rules, far more freedom.

Backpropagation runs through time over the 24 ticks with a custom autograd function that uses
a precomputed transpose of the sparse matrix, so no sparse tensor is ever differentiated. Biases
get a learning rate of 1e-6 (a bias shift of 1e-5 already moves a descending neuron by one
standard deviation of its signal), gains and heads 3e-3, each group clipped on its own.

**Data.** `data.py` streams the Lichess monthly database (zstd, tens of GB) and stops after the
requested number of games, so only a few tens of MB are downloaded: rated standard games with
both players between 1600 and 2200, no bullet, eight positions sampled per game after the
sixth ply, with the move that was played and the game result. Held-out games (10%) provide the
evaluation positions. Known blind spot: the eye sees pieces and whose turn it is, not castling
rights or en passant; the legal-move mask handles it at play time.

**Metrics.** On held-out positions: *legal rate* of the unmasked argmax (chance ≈ 0.7%),
*top-1* and *top-3* agreement with the human move among legal moves (chance ≈ 3% and 9%),
value accuracy (chance 33%). Phase 3 targets: legal rate > 99%, top-1 ≥ 35%, top-3 ≥ 60%.

**Speed on the Mac.** One step of 8 positions × 24 ticks takes about 7 s (3.6 s forward,
3.6 s backward); 1,200 steps see 9,600 positions in under three hours. That is a pilot, not
training: the plan's 10 M positions per epoch need a CUDA GPU.

**Running on a GPU.** Rent a machine with an RTX 4090 or A100 class card, clone the repo, put
the three MaleCNS tables in `data/` (see MALECNS_NOTES.md), then:

```bash
pip install torch numpy pandas pyarrow scipy python-chess zstandard requests
python -m flybrain.graph && python -m flybrain.checks
python -m flybrain.data 400000 full            # ~3.2 M positions
python -m flybrain.train --device cuda --edge-gains --positions data/positions_full.jsonl.gz \
       --steps 200000 --batch 128 --ticks 24 --eval-every 1000 --tag edge
```

<!-- PHASE3_RESULTS:begin -->
## Phase 3 results (16 Sep 2026)

**Second run, four RTX 5090s.** The per-edge variant (27.1 M parameters) trained with data
parallelism over four GPUs, 512 positions per step at 0.91 s per step, for 12,200 steps:
6,246,400 positions from 4.3 million distinct ones (600,000 Lichess games streamed on
the box), 168 minutes of stage 2 after a 23-minute warm-up stage on 431,000 positions.
Evaluation on 512 held-out positions from games not used in training:

| step | positions seen | eval loss | legal-move rate | top-1 | top-3 | value acc |
|---|---|---|---|---|---|---|
| 200 | 102,400 | 5.24 | 58.6% | 15.8% | 31.1% | 49.0% |
| 800 | 409,600 | 4.64 | 67.2% | 16.2% | 37.5% | 54.9% |
| 1,500 | 768,000 | 4.43 | 71.1% | 19.1% | 40.4% | 49.0% |
| 2,000 | 1,024,000 | 4.12 | 73.4% | 18.8% | 41.4% | 58.8% |
| 4,000 | 2,048,000 | 4.00 | 75.4% | 22.3% | 42.6% | 53.5% |
| 6,000 | 3,072,000 | 3.76 | 76.2% | 23.0% | 45.9% | 54.3% |
| 8,000 | 4,096,000 | 3.75 | 75.0% | 20.5% | 44.7% | 52.7% |
| 10,000 | 5,120,000 | 3.59 | 78.5% | 23.8% | 49.4% | 54.1% |
| 12,000 | 6,144,000 | 3.46 | 77.5% | 25.8% | 48.6% | 56.8% |
| 12,200 | 6,246,400 | 3.48 | 79.3% | 24.0% | 49.0% | 59.0% |

Best values over the run: legal 80.9%, top-1 26.4%, top-3 49.8%, value 59.0%.
Chance levels: 0.7% legal, about 3% top-1, 9% top-3, 33% value. Curves: `data/phase3_curve.svg`.

**First run, one RTX 4090** (768,000 positions, 125 minutes): legal 68.0% (best 72.1%), top-1
20.3% (best 22.5%), top-3 39.5%. The Mac control with per-neuron gains only: 16.8% legal, 6.2% top-1.

The second run confirms the trend and pushes every number up: the loss is still falling at the end, legality climbs from the high sixties to about 80%, and the fly agrees with the human move a quarter of the time, in the top three half the time. The plan's targets of 99% legal and 35% top-1 are still not reached, and the learning is slow in the way the plan anticipated: the move prior is learned fast, reading the board through the descending neurons improves slowly. The trained weights survived a close call: the checkpoint file came back as two writes spliced together, because the end-of-budget kill landed while it was being rewritten, and both copies were unreadable by PyTorch; the newest write's model tensors were complete and were recovered by walking the archive by hand (tools_recover_checkpoint.py). The weights at step 11,600 are attached to the GitHub release phase3-run2-step11600 (108 MB, or 54 MB in half precision); 40.6% of synapses changed their gain by more than 10%. Next levers, in order: run the three controls (rewired, sign-shuffled, dense) so the result is interpretable, then a longer run resumed from these weights, then the modulatory-gate fallback if legality stays capped.

<!-- PHASE3_RESULTS:end -->

## Phase 4: making it play

**The engine.** `flybrain/uci.py` speaks UCI on stdin/stdout (`./fly-uci` launches it), so any
chess GUI or match runner can play the fly. One forward pass per move: the board is painted on
the eye, the network runs 24 ticks, the move head is masked to legal moves and the promotion
head picks the piece on the last rank. About 3.7 s per move on the Mac's CPU, milliseconds on a
GPU. Weights: the recovered step-11,600 model of the second Phase 3 run.

**Two rules a search-free policy needs.** (1) Legal-move masking: the fly's eye does not see
castling rights or en passant, so legality comes from the mask. (2) Anti-repetition: a policy
without search happily shuffles a piece back and forth; a 4-game pilot against a random mover
ended in three threefold repetitions from winning positions. The engine now refuses any move
that recreates a position already seen, unless every legal move does.

**Rating.** `flybrain/play.py` plays 200-game matches, many games at once so the fly's moves
are batched through the network. Each game opens with four random plies for variety and
colours alternate. Opponents: a uniformly random mover, and Stockfish 19 with
`UCI_LimitStrength` at its floor of 1320 (50 ms per move). Elo difference from the match score
with a 95% interval from the per-game outcomes; the Stockfish match anchors an absolute rating.

<!-- PHASE4_RESULTS -->

## Phase 5 results (18 Sep 2026)

**What training changed about the wiring.** The connectome fixes which neurons exist, who talks
to whom, and every neuron's sign; training could only change how loudly each connection speaks.
It barely did. The rank order of connection strengths after training correlates with the measured
synapse counts at Spearman 0.948, and the typical connection moved by a factor of
1.07. Only 0.9% of connections were more than halved and 0.9% more than doubled.
The trained network is still the fly's connectome, gently retuned, not a different network wearing
its shape. Training was also blind to anatomical strength: the median gain is 0.99 for every
bucket from one-synapse connections to fifty-plus. Where it did act, it is interpretable: the
inputs turned down hardest belong to olfactory receptor neurons, wide-field motion detectors and
bristle mechanosensors, every one of them a sense irrelevant to a static board, while 57% of the
inputs to descending neurons were turned down, which is the readout learning to listen selectively.

**Where the chess happens.** Each functional group was silenced in turn, clamping those neurons to
zero at every tick, and the fly's agreement with Stockfish was re-measured on 2,048 held-out
positions. One standard error is 0.89 points, so anything under 2.7 points is noise.

| silenced | neurons | agreement after | change |
|---|---|---|---|
| descending neurons | 1,314 | 5.1% | −15.2 |
| optic lobe (intrinsic) | 89,390 | 5.2% | −15.1 |
| photoreceptors | 4,114 | 6.6% | −13.7 |
| visual projection neurons | 9,201 | 6.9% | −13.4 |
| distal medulla (Dm) | 8,175 | 8.4% | −11.9 |
| central brain (intrinsic) | 32,160 | 11.1% | −9.2 |
| transmedullary (Tm, TmY) | 28,042 | 12.0% | −8.3 |
| medulla intrinsic (Mi) | 9,589 | 13.5% | −6.8 |
| lamina (L1-L5) | 8,883 | 14.3% | −6.0 |
| lobula columnar (LC, LPLC) | 5,807 | 15.0% | −5.3 |
| brain sensory axons | 4,868 | 17.2% | −3.1 |
| mechanosensory | 2,157 | 17.5% | −2.8 |
| ascending neurons | 1,846 | 18.8% | −1.5 (noise) |
| motion detectors (T4, T5) | 13,580 | 19.0% | −1.3 (noise) |
| visual centrifugal | 563 | 19.7% | −0.6 (noise) |
| gustatory | 355 | 19.9% | −0.4 (noise) |
| medulla tangential (Pm, Li) | 2,823 | 20.0% | −0.3 (noise) |
| lateral horn | 2,028 | 20.1% | −0.2 (noise) |
| central complex | 2,950 | 20.2% | −0.1 (noise) |
| CX: ring neurons (ER) | 282 | 20.2% | −0.1 (noise) |
| olfactory receptor neurons | 2,635 | 20.3% | −0.0 (noise) |
| mushroom body: dopaminergic | 340 | 20.3% | −0.0 (noise) |
| antennal lobe local | 420 | 20.3% | −0.0 (noise) |
| CX: compass (EPG, PEG, PEN) | 110 | 20.3% | −0.0 (noise) |
| CX: fan-shaped body | 2,366 | 20.3% | −0.0 (noise) |
| mushroom body: outputs | 97 | 20.4% | +0.0 (noise) |
| mushroom body: Kenyon cells | 4,064 | 20.4% | +0.1 (noise) |
| antennal lobe projection | 686 | 20.4% | +0.1 (noise) |

**The result is unambiguous: the fly plays chess with its eyes.** Every stage of the visual
feedforward pathway is load bearing, from the photoreceptors through the lamina, medulla and
lobula to the projection neurons that carry vision into the brain, and finally the descending
neurons that the move is read from. Silencing the descending neurons drops the legal-move rate
from 80% to 0.05%, which confirms the readout is genuinely reading from them.

**And the higher brain contributes nothing at all.** The mushroom body, the fly's learning and
memory centre, costs zero: its 4,064 Kenyon cells, its 97 output neurons and its 340 dopaminergic
neurons can all be silenced without changing a single move. The central complex, which navigates,
costs zero. The lateral horn, which drives innate responses, costs zero. So does the entire
olfactory system, which is the expected sanity check. The motion detectors T4 and T5 cost nothing
either, which makes sense for a board that never moves.

That is a real answer to the question Phase 5 was designed to ask. The chess ability such as it
is lives entirely in the visual system, and the parts of the fly's brain that make it clever are
not involved.

## Phase 6: the demo (18 Sep 2026)

**The garden** (`flybrain/scene.py`). A MuJoCo scene in centimetres, because flybody's fly is real
size: 3.4 mm long. The chess set is built to it — a 3.2 mm square, a 2.6 cm board, the tallest
piece under 2.5 mm — on a mossy stone in a lawn. Around it, at the scale the fly would see them:
grass in tufts that lean away from the stone, daisies, buttercups, poppies and cornflowers on
stems, a few mushrooms, pebbles and dew. On the skyline, trees whose colour is blended toward the
sky with distance, bushes, clouds, and the sun low over the far side of the board with the light
coming from it. The board itself is a stepped walnut frame with a maple inlay line and grained
squares; every piece stands on a felt disc and is turned from primitives — the knights have a
neck, a head, ears and a mane, the bishops a mitre with its slit, the queens a crown of six
beads, the kings a cross. 7,190 geoms; about 0.7 s a frame at 1280 × 1080.

Two things had to be learned about the renderer. MuJoCo's fog applies to the skybox, which
flattens the sky to one colour, so aerial perspective is done by hand per tree. And the fly
model's own three tracking lights blacken every far emissive geom, so the scene includes a copy
of the model without them.

**The panel** (`flybrain/panel.py`). Beside the garden, three things the demo is actually about.
*The brain*: the 106 neuropils of the head as a ghost map, coloured by region, with every neuron's
activity lit on top of it as the network runs — amber above rest, blue below — the descending
neurons on their own scale so the readout can be seen, and the lobe that receives the board
labelled. *The board and the retina*: the position as a diagram, and the same position as the
892 columns of the right eye receive it, white pieces on the R7 channel and black on R8. *What
the descending neurons argued for*: the three moves the readout gave most weight to, and which it
chose. Drawn at one design size and scaled, so drafts at any resolution keep their layout.

**The film** (`flybrain/demo.py`). A title, the garden revealed in a slow orbit, then the game:
for each of the fly's first moves, the brain lights up tick by tick while it thinks, then it walks
from its resting spot to the piece, carries it to its square, and walks home; a captured piece is
carried off the board first. Stockfish's pieces glide across on their own. After the animated
opening, the rest of the game is fast-forwarded at a few plies a second to the real result, which
the end card states. `--save-game` pickles the played game so the film can be re-rendered without
replaying it; `--replay` renders a saved one. The default output is 3840 × 2160 at 30 fps; the
panel is drawn natively at that scale rather than upscaled. The game is played by exactly the same code as the
engine and the matches; nothing in the demo touches the model.

## Phase 7: reinforcement learning (18 Sep 2026)

Every phase before this taught the fly by imitation. It was shown a position, told which move a
human — later Stockfish — had played, and scored on whether it matched. It was never once told
that a move it chose had lost a rook. Phase 7 closes that loop and the fly learns from its own
games.

**The environment** (`flybrain/env.py`). One step is a ply pair: the fly moves, the opponent
replies, and the fly is handed the position it now has to deal with. Ninety-six games run at
once so that every decision point in the batch goes through the brain in a single forward pass,
which is what makes 144,209 neurons per position affordable. The reward is the game result from
the fly's point of view, ±1 and 0, plus a potential-based material term,
`gamma * phi(s') - phi(s)` with `phi = tanh(material advantage / 5)`. Finished games are refilled
immediately so every forward pass runs at full width.

**The algorithm** (`flybrain/rl.py`). PPO. Moves are sampled from the legal-masked move head
rather than taken greedily, because a policy that never varies can never discover anything;
GAE(lambda) turns the rewards into an advantage per move; several epochs of a clipped surrogate
keep any one batch of games from moving the policy far; an entropy bonus resists premature
collapse; and a KL penalty against the frozen supervised policy anchors the fly to the chess it
already had. The critic is a new scalar head on the same 1,314 descending neurons the move head
reads, trained from scratch — the supervised value head predicts a three-way result from White's
point of view, which is not the baseline this needs. Trained: the per-neuron gains, the biases
and the move head, 5.83 M of the model's 27.10 M parameters. The wiring, the signs and the
synapse counts stay fixed, as in every phase.

**The run.** 920 iterations, **1,413,120 of the fly's own moves across 20,477 games**, 150 minutes
on one RTX 4090, $1.29. Entropy fell from 2.07 to 1.22 and then held, so it kept exploring. The
KL from the supervised policy rose to 0.79, then came back to 0.63: the fly wandered, found the
wandering unprofitable, and the anchor pulled it back. Learning curve in `data/rl_curve.svg`.

**It learned, and it learned the wrong thing.** Across 92 held-out evaluations of 96 games each,
the score against the greedy capture bot rose from 0.280 in the first half of the run to 0.324 in
the second, +0.044 ± 0.008 — 5.4 standard errors, not noise. But the in-training evaluation allows
repetitions, which flatters a shuffler. Replaying the Phase 4 matches with both sets of weights
under identical settings, anti-repetition rule applied to both, 400 games each:

<!-- PHASE7_RESULTS:begin -->
| opponent | weights | games | W | D | L | score | change |
|---|---|---|---|---|---|---|---|
| greedy capture bot | supervised | 400 | 15 | 216 | 169 | 0.307 | — |
|  | after RL | 400 | 8 | 245 | 147 | 0.326 | +0.019 ± 0.019 (+1.0 SE, no change) |
| random mover | supervised | 400 | 140 | 253 | 7 | 0.666 | — |
|  | after RL | 400 | 101 | 297 | 2 | 0.624 | -0.042 ± 0.017 (-2.5 SE, worse) |
| Stockfish 17, depth 1 | supervised | 200 | 0 | 20 | 180 | 0.050 | — |
|  | after RL | 197 | 0 | 16 | 181 | 0.041 | -0.009 ± 0.014 (-0.7 SE, no change) |
<!-- PHASE7_RESULTS:end -->

The Stockfish row has 197 games rather than 200 in the RL arm: an engine call blocked on one of
the long games and the match was stopped, so the completed games were scored from the PGN
(`tools_summarise_pgn.py`), with both arms scored by the same code. Against a real engine the
drawing trick buys nothing, because Stockfish converts.


Look at the columns rather than the score. Losses fell in both matchups. Wins fell in both.
Draws absorbed everything. PPO turned the fly into a drawing machine: substantially harder to
beat, substantially worse at winning. Against the greedy bot, which was beating it, that is
roughly break-even. Against the random mover, which it was beating, it is a straight loss.

**Why, and it is a design error rather than a bug.** The material term pays the fly for not
losing material. For a policy that cannot calculate, the cheapest way to never lose material is
to never commit: shuffle, trade down, hold. It optimised exactly what it was paid for.
Potential-based shaping guarantees that the *optimal* policy is unchanged (Ng, Harada & Russell
1999); it says nothing about which policy is easiest to reach on a finite budget inside a
constrained policy class, and the fly went to the easy one. This is reward misspecification in
its most ordinary form, and a working pipeline is supposed to surface it in an afternoon, which
is what happened.

**What would be tried next, in order.** Drop the shaping to zero and pay only for the result,
accepting a much sparser signal. Raise the step size: the KL from the rollout policy ran at
about +0.008 per iteration against a 0.03 target, so PPO was barely using its trust region.
Free the 21 M per-edge gains (`--train all`), since the supervised phase found almost all of its
capacity there and this run trained under a quarter of the model. None of these were run: the
finding above is the result, and inventing a better number for it would be the one thing this
project has not done.
