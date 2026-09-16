# flybrain — the digital fly (Phases 1–3)

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

The second run confirms the trend and pushes every number up: the loss is still falling at the end, legality climbs from the high sixties to about 80%, and the fly agrees with the human move a quarter of the time, in the top three half the time. The plan's targets of 99% legal and 35% top-1 are still not reached, and the learning is slow in the way the plan anticipated: the move prior is learned fast, reading the board through the descending neurons improves slowly. One operational loss: both copies of the final checkpoint came back corrupt through the rental host's SSH proxy (identical truncated size twice), and the instance was destroyed when the credit ran out, so the trained weights of this run are gone; the logs are complete. Next time the box verifies a checksum before the copy and ships a 55 MB half-precision model-only file. Next levers, in order: run the three controls (rewired, sign-shuffled, dense) so the result is interpretable, then a longer run from a fresh checkpoint, then the modulatory-gate fallback if legality stays capped.

<!-- PHASE3_RESULTS:end -->
