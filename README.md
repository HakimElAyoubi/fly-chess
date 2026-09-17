# Fly Chess

Teaching the fruit fly's wiring diagram to play chess.

A living fruit fly cannot play chess. Its complete wiring diagram, the MaleCNS connectome
published by Janelia, Cambridge and Google Research in September 2026, can be made to try.
This project keeps every neuron, every connection and every connection sign exactly as
measured, learns only the connection strengths the connectome does not measure, shows the
board to the fly's photoreceptors, and reads its move from the descending neurons that carry
every command from the real fly's brain to its body.

The full plan is in [PLAN.md](PLAN.md) (six phases). Status:

| phase | | status |
|---|---|---|
| 0 | Know the data | done |
| 1 | Build the digital fly | done |
| 2 | Show it the board | done: the medulla reads 100.0% of squares |
| 3 | Teach it the rules | in progress: 79.3% legal, 24.0% top-1 after 6.2 M positions |
| 4 | Make it play | in progress: UCI engine built, rating matches running |
| 5 | Look inside | |
| 6 | The demo: a fly-scale chess set in a garden, the fly walks over and moves the pieces | |

## What is here

- `viewer.html` — the Cloud Atlas: an interactive 3D point cloud of all 108 neuropils of the
  male fly's brain and nerve cord, 140,024 neuron cell bodies, plain-language notes on every
  region, and playback of the digital fly's simulated activity. Open it in any browser.
- `flybrain/` — Phase 1: the connectome-constrained brain model (144,209 neurons, 21.27 M
  signed connections), the two biology checks it passes, and the activity export. See
  [flybrain/README.md](flybrain/README.md) for the model, the results and its known limitation.
- `build_cloud.py`, `build_viewer.py`, `viewer_template.html` — how the atlas is built.
- `MALECNS_NOTES.md` — reference notes on the dataset: papers, numbers, bucket layout, schemas.
- `data/` — small derived results (`checks.json`, `spectrum.json`, `graph_brain_summary.json`,
  `activity.json`, `cloud.json`). The raw tables are not committed; they are public and the
  notes say where to get them.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install cloud-volume numpy pandas pyarrow scipy torch
# fetch the three v1.0 tables listed in MALECNS_NOTES.md into data/, then:
python -m flybrain.graph            # signed sparse matrix of the brain
python -m flybrain.checks           # biology checks + gain calibration
python -m flybrain.export_activity  # activity frames for the atlas
python build_viewer.py              # -> viewer.html
```

## Data and credit

All connectome data is MaleCNS v1.0 (Berg et al., *Cell*, 2026), released under CC-BY by the
Janelia FlyEM Project Team, the Drosophila Connectomics Group at Cambridge and Google Research:
https://male-cns.janelia.org. The modelling approach follows Shiu et al. (*Nature*, 2024) and
Lappalainen et al. (*Nature*, 2024).

Hakim El Ayoubi, 2026.
