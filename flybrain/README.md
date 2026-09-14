# flybrain — the digital fly (Phases 1–2)

A connectome-constrained rate model of the male fruit fly brain, built from MaleCNS v1.0.

| file | what it does |
|---|---|
| `graph.py` | builds `data/graph_brain.npz` (CSR, W[post, pre] = synapse count × sign of pre) and `graph_brain_meta.feather` |
| `model.py` | `FlyBrain`: r ← r + k·(f(αWr + I + b) − r); tanh / threshold / mixed units; input normalisation |
| `checks.py` | the two biology checks and gain calibration → `data/checks.json` |
| `export_activity.py` | tick-by-tick activity of both experiments → `data/activity.json` (injected into the atlas by `build_viewer.py`) |
| `eye.py` | the right eye as a screen: column map, photoreceptor placement, chessboard → photoreceptor currents |
| `probe.py` | Phase 2 probe: random positions through the model, linear readout of the board from each stage → `data/probe.json` |

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

**The probe.** 4,096 uniformly random placements (each square empty with p = 0.4, else a
uniform piece; not legal chess, deliberately, so every state appears on every square hundreds
of times) plus 1,024 positions from random play as a realistic test set. Training uses 3,072
placements, 512 choose the regularisation, 512 are held out. Each position runs through the
untrained Phase 1 model for 24 ticks; the steady-state activity of each stage is read by a
linear classifier (PCA to at most 2,048 components, one softmax regression head per square
with 13 classes and one for the side to move, fit to convergence with L-BFGS). A least-squares
readout on one-hot targets is reported alongside; it masks classes whose codes are collinear.

**Results** (5,120 positions, 3,072 for training, 24 ticks, gain 1.5; a readout that
always answers "empty" scores 40% on random placements and about 44% on random play):

| stage | neurons | responding | square accuracy, random placements | square accuracy, random play | side to move |
|---|---|---|---|---|---|
| Photoreceptor input (the stimulus itself) | 2,239 | 97.5% | **98.8%** | 99.7% | 100.0% |
| Lamina L1–L5, right | 4,466 | 88.9% | **90.0%** | 94.0% | 100.0% |
| Medulla + lobula, right (all optic-lobe intrinsic) | 44,789 | 75.5% | **91.6%** | 95.5% | 100.0% |
| Visual projection neurons, right | 4,612 | 41.7% | **87.3%** | 92.8% | 100.0% |
| Central brain intrinsic | 32,160 | 0.9% | **55.8%** | 67.4% | 100.0% |
| Descending neurons | 1,314 | 0.5% | **47.4%** | 61.2% | 100.0% |

Per piece, from the medulla on random placements: empty 100.0%, then between 78.2% (white knights)
and 96.3% (black kings). The side to move is read perfectly from every stage.

**Verdict: target not met yet.** The plan asks for more than 99.5% of squares from the medulla
and lobula; the untrained optic lobe gives 91.6% (95.5% on realistic positions). The
stimulus is not the limit: the same readout recovers 98.8% from the
photoreceptor currents and a per-square classifier 100%. The loss happens inside the network:
tanh units compress the level code and lateral connections mix neighbouring squares in a way a
linear readout cannot fully undo. Fixing the stimulus once already helped a lot (with currents
2.0 × level the medulla read only 74.6%); two more levers remain before anything in the wiring
is touched, each one 25-minute run: cap the receptor rates at 0.5 instead of 0.8 so the
medulla stays in its linear range, and double the probe's training set. The descending neurons
already carry the board at 47.4% (61.2% realistic) before any training, the baseline Phase 3
must beat.

