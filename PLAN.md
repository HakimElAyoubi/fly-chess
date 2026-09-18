# Chess for the Digital Fly — project plan

Compiled 2026-09-14. Designed version: plan.html (published as an artifact).

## The honest version

A living fruit fly cannot play chess. It learns simple associations over minutes and each eye
gives it ~750 image points; there is no route to perceiving 64 squares × 12 piece kinds, let
alone planning. The nearest real-fly experiment is in the appendix and is a performance piece.

What can be done: train the **digital fly**. The connectome fixes every neuron, every
connection and (via neurotransmitter predictions) the sign of each connection. It does not
measure connection strengths. We keep the anatomy fixed, learn only the strengths, show the
board to the photoreceptors, read the move from the 1,314 descending neurons, and ask whether
that wiring can support chess.

Precedents (both Nature 2024): Shiu et al. simulated the whole female brain from a connectome
and reproduced feeding/grooming circuits; Lappalainen et al. task-trained a connectome-
constrained network of the optic lobe and it predicted real neural responses (code: flyvis).

End claim: "the male fly's wiring, with learned strengths, plays at Elo X and these regions do
the work" — or the honest negative: "the wiring can't carry it, and here is where it fails."

## Rules of the game

| Fixed by the connectome | Learned |
|---|---|
| neuron set (all traced brain neurons) | non-negative gain per existing connection |
| edge set (25.6 M, none added) | bias + time constant per cell type |
| sign per neuron (ACh +, GABA −, Glu −) | linear readout DN rates → move |
| initial strength = synapse count | |

Not allowed: new neurons/edges, flipped signs, input anywhere but photoreceptors, output from
anywhere but descending neurons.

Controls (same data, same compute): degree-matched random rewiring; shuffled signs; a plain
dense net of similar parameter count.

Signal path: Board (8×8, 13 states) → eye (~750 columns) → optic lobe (89,403 neurons) →
central brain (~41,000) → descending neurons (1,314) → 4,096 from–to logits + 4 promotions,
masked to legal moves at play time. Recurrent, 32 ticks per move.

## Phases

### 0 — Know the data (done)
Notes (MALECNS_NOTES.md), annotation table in data/, the Cloud Atlas viewer.

### 1 — Build the digital fly (weeks 1–2) — DONE 2026-09-14 (see flybrain/README.md)
Result: 144,209 neurons, 21.27 M signed edges; graded input-normalised model at gain 1.5 passes
both checks (sugar→MN9 z = 4.8 vs 20 controls; light drives only the stimulated eye's motion
pathway). Caveat: input normalisation makes the motor response small (~1% of peak); learned
gains (Phase 3) are the fix. CPU only: 0.7 s per tick at batch 128; MPS has no sparse support.
1. Fetch connectome-weights (traced-only, 508 MB) and body-neurotransmitters (43 MB) from
   gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/.
2. Neuron set: Traced, brain side (ol_*, cb_*, visual_*, descending, ascending, brain sensory).
   Drop the VNC for v1. ~130k neurons, ~20 M edges.
3. Sparse W (post × pre) = synapse count × sign of presynaptic transmitter. DA/5-HT/OA = 0 in v1.
4. Rate model: r ← r + (dt/τ)(−r + softplus(W·r + b + I)); dt 5 ms, τ ~20 ms, 32 ticks/move.
5. PyTorch CSR sparse matmul, batched, BPTT, gradient clipping; spectral scaling at init.
6. Biology checks before chess: sugar GRNs → proboscis motor neurons in GNG respond (Shiu
   et al.); one eye stimulated → lobula plate motion cells respond.
7. Atlas activity mode: 140k cell bodies glow by rate, tick by tick.
Done when: 130k-neuron forward pass at batch 128 well under 1 s on GPU; both checks pass;
activity renders.

### 2 — Show it the board (week 3) — DONE 2026-09-15 (see flybrain/README.md)
<!-- RESULTS:begin -->
Result: target met. Retinotopic linear readout of the untrained medulla + lobula: 100.0% of
squares on random placements (100.0% realistic); generic principal-component readout 99.98% / 100.0%
(the stimulus itself reads at 100.0%). Descending neurons 50.1% / 63.5% before
any training. 7,168 training boards; receptor-rate cap left at 0.8 (0.5 and 0.3 made no difference).
Table and details in flybrain/README.md.
<!-- RESULTS:end -->
1. Eye map from annotation columns assignedOlHex1/assignedOlHex2: 892 columns in the right eye.
   Photoreceptors placed by their strongest one-per-column partner. The board is a retinotopic
   quantile grid (8 rank bands × 8 files of equal column count) over the 645 columns that have
   both colour receptors traced: 10 columns per square. (The 3×3-facet patch of the original
   plan was not possible: only 380 columns have traced R1–R6 cells.)
2. Square state on the R7/R8 colour channels as a point on a 4×4 grid of levels (empty (0,0),
   white pieces along the R7 edge, black along the R8 edge); side to move = global brightness
   bias on R1–R6. Pieces are told apart by colour, not shape: the eye cannot resolve glyphs.
3. Injected as photoreceptor currents into 2,239 right-eye photoreceptor axons (current 2.0 × level).
4. Linear probe from medulla+lobula at the last tick must recover the board > 99.5% per square
   with untrained gains. Probe DNs too (baseline).
Done when: the optic lobe provably carries the whole board.

### 3 — Teach it the rules and good moves (weeks 4–6) — IN PROGRESS (first GPU run 2026-09-15)
Pipeline: flybrain/data.py (Lichess stream), policy.py (eye → network with learned gains → DN readout),
train.py (imitation loss, legality/top-1/top-3 metrics, checkpoints, progress bar). Mac pilot running
with per-neuron gains; the per-edge variant is one flag away and needs a rented GPU.
<!-- PHASE3_RESULTS:begin -->
GPU runs: (1) one RTX 4090, 768k positions: legal 68.0%, top-1 20.3%; (2) four RTX 5090s,
6.2 M positions: legal 79.3% (best 80.9%), top-1 24.0% (best 26.4%),
top-3 49.0% (best 49.8%), eval loss 3.48. Mac control (per-neuron gains): 16.8% / 6.2%.
The second run confirms the trend and pushes every number up: the loss is still falling at the end, legality climbs from the high sixties to about 80%, and the fly agrees with the human move a quarter of the time, in the top three half the time. The plan's targets of 99% legal and 35% top-1 are still not reached, and the learning is slow in the way the plan anticipated: the move prior is learned fast, reading the board through the descending neurons improves slowly. The trained weights survived a close call: the checkpoint file came back as two writes spliced together, because the end-of-budget kill landed while it was being rewritten, and both copies were unreadable by PyTorch; the newest write's model tensors were complete and were recovered by walking the archive by hand (tools_recover_checkpoint.py). The weights at step 11,600 are attached to the GitHub release phase3-run2-step11600 (108 MB, or 54 MB in half precision); 40.6% of synapses changed their gain by more than 10%. Next levers, in order: run the three controls (rewired, sign-shuffled, dense) so the result is interpretable, then a longer run resumed from these weights, then the modulatory-gate fallback if legality stays capped.
<!-- PHASE3_RESULTS:end -->
1. Data: Lichess DB, ≥ 10 M positions (1600–2200), Stockfish depth-10 labels for 1 M; python-chess.
2. Readout: linear, 1,314 DN rates → 4,096 + 4 logits; unmasked in training, masked at play.
3. Loss: cross-entropy on target move + value head (W/D/L). AdamW, clipping, T = 32.
   ~2–10 h per 10 M-position epoch on an RTX 4090-class GPU; 5–10 epochs.
4. Curriculum: legality (> 99% legal unmasked) → imitation (top-1 ≥ 35%, top-3 ≥ 60% held-out;
   Maia-class nets reach ~50% top-1) → value head.
5. Run the three controls with identical budgets.
Done when: > 99% legal, top-1 ≥ 35%, controls table filled.

### 4 — Make it play (weeks 7–10) — IN PROGRESS (engine built 2026-09-17)
flybrain/uci.py (UCI engine, ./fly-uci), flybrain/play.py (batched matches vs random / Stockfish 1320,
Elo with 95% CI). Anti-repetition rule added after a pilot drew won games by repetition.
<!-- PHASE4_RESULTS -->
1. UCI engine wrapper (python-chess), one forward pass per move (~10 ms GPU, ~1 s Mac CPU).
2. RL vs Stockfish UCI_LimitStrength 1350 → 1600 → 1800: REINFORCE with value baseline or
   DAgger; keep imitation loss mixed in.
3. Rate with cutechess-cli, 200-game matches, ordo Elo with error bars. Targets: 100% vs random;
   ≥ 1200 vs Stockfish-limited. Expectation 1000–1500; < 1000 means the constraint binds.
Done when: engine binary with a measured rating.

### 5 — Look inside (weeks 11–12) — DONE 2026-09-18
Result: the fly plays chess with its eyes. Silencing each functional group and re-measuring agreement
with Stockfish on 2,048 positions: the whole visual feedforward pathway is load bearing (photoreceptors −13.7,
optic lobe −15.1, distal medulla −11.9, transmedullary −8.3, lamina −6.0, lobula columnar −5.3 points) as are
the descending neurons (−15.2, and the legal rate collapses to 0.05%). The higher brain contributes nothing:
mushroom body Kenyon cells, output neurons and dopaminergic neurons, the central complex and the lateral horn
are all within noise of zero. Training also barely changed the wiring: Spearman 0.948 against the measured
synapse counts, typical connection moved 1.07x. Details in flybrain/README.md.
1. Ablate each of 108 neuropils; re-measure accuracy/Elo; heat map in the atlas.
2. Decode across ticks: where/when are from-square, to-square, check first readable?
3. Learned gains vs synapse counts: near 1 → anatomy did the work; divergent → rewired within
   the constraint.
Done when: ablation map, decoding timeline, gains-vs-anatomy figure.

### 6 — The demo: a fly playing chess in a garden (weeks 13–15)
Goal: a garden scene with a chess set built at fly scale, where the fly walks over and moves the
pieces. Opponent: a human or a light Stockfish.

**The boundary, decided 2026-09-17.** Exactly one thing is computed: the fly seeing the board and
choosing a move. Board state is painted on the real retinotopic map of its right eye, runs 24
ticks through 144,209 neurons and 21.27 M connections, and the move is read from the 1,314
descending neurons. Everything after that is animation. Walking, gripping, carrying and returning
are performed, not simulated. No motor learning, no physical control, no contact tuning.

**What this removes.** The whole motor RL problem, the 108-degree-of-freedom control problem, and
the speed problem (flybody runs at 0.3x realtime only when its full articulated dynamics are
simulated, which now they never are). Chess RL stays headless as before, and nothing about the
demo constrains it.

1. **The asset.** `flybody` (Google DeepMind and Janelia, Nature 2025) used as a model, not as an
   agent: anatomically correct mesh, veined wings, compound eyes, jointed tarsi, built from real
   fly measurements by the institute that produced the connectome. Positioned directly each frame.
2. **The walk cycle.** Recorded once from flybody's own trained locomotion controller, then looped
   and steered along the path, so the gait is a real fly's tripod gait rather than something
   hand-drawn.
3. **The set, built for the fly.** A jewel-sized board a few centimetres across with pieces of a
   few milligrams, on a stone in the grass. A real-scale fly on a chess set made for it.
4. **The performance of a move.** From the resting spot to the source square, grip, carry to the
   destination, release, walk home to exactly where it started. A captured piece is carried off
   the board first.
5. **The thinking pause is the feature.** A move costs a second or two of real computation. Fill
   it with the Cloud Atlas inset lighting up as activity crosses the brain, so the one honest
   moment in the demo is also the most interesting thing on screen.
6. **The opponent.** A person clicking a piece, or Stockfish at limited strength through the
   engine built in Phase 4.
Done when: a stranger can play the fly in a garden and watch it walk over and move the pieces.

**Open fork.** With no physics left, the choice of MuJoCo versus the browser is now about who
watches it. MuJoCo uses the flybody meshes natively and renders a better hero video; a browser
build is a link anyone can open and reuses the Cloud Atlas viewer for the brain inset. The
flybody meshes export to glTF, so the asset survives either choice.

## Compute and tools

| Need | Choice | Notes |
|---|---|---|
| prototyping, data prep, engine, viewer | Mac, Apple M4, 16 GB | fine for the graph (~250 MB) and small batches; not for training |
| training (phases 3–5) | 1 rented GPU (4090/A100) | ~50–150 GPU-h, ~$50–300 |
| model | PyTorch, torch.sparse CSR; flyvis as reference recipe | |
| chess | python-chess, Stockfish, cutechess-cli, ordo | |
| positions | Lichess open database | |
| connectome | gs://flyem-male-cns v1.0 (public, CC-BY) | |

## Risks and fallbacks

- Constraint too tight → per-edge gains + biases; then DA/5-HT/OA edges as learned
  multiplicative gates (how the mushroom body actually learns); last resort a small % of new
  edges, reported as a deviation.
- Recurrent net explodes → spectral radius < 1 at init, clipping, 16 ticks, tanh.
- Eye can't resolve the board → Phase 2 probe gates everything; enlarge to 4×4 facets, both eyes.
- Compute creeps → fewer positions, 16 ticks, mixed precision, per-type-pair gains (~11k types)
  before per-edge (20 M).
- "Not a real fly" → rules of the game, biology checks, controls table.

## Appendix — the living-fly option

Fly in the loop: tethered fly on an air-supported ball in closed-loop VR (FicTrac + LED
arena). The engine proposes two candidate moves as two patterns left/right; the fly's turning
picks one. Operant conditioning with IR heat (the Heisenberg flight-simulator paradigm) teaches
a pattern preference in an hour or two. Result: a fly that picks the cued pattern, i.e. the
engine's move, not a fly that plays chess. Needs a fly lab and ~$10–30k of rig. Performance
piece only, and only after the digital fly has a rating to compare against.

## Sources

- Berg et al., Cell 2026 (MaleCNS). https://male-cns.janelia.org/
- Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 634, 2024.
- Lappalainen et al., "Connectome-constrained networks predict neural activity across the fly visual system", Nature 634, 2024. https://github.com/TuragaLab/flyvis
- McIlroy-Young et al., Maia chess, 2020.
- python-chess, Stockfish, cutechess, Lichess database.
