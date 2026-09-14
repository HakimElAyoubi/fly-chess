# Male CNS (MaleCNS) connectome — project reference notes

Compiled 2026-09-14 from natverse.org/malecns, male-cns.janelia.org, the Google Research blog,
the Cell paper (Berg et al. 2026) and a direct inspection of the public GCS bucket.

## What it is
- Complete synaptic wiring diagram of the central nervous system (brain + optic lobes + ventral nerve cord)
  of ONE adult male Drosophila melanogaster.
- Producers: Janelia FlyEM Project Team (HHMI), Drosophila Connectomics Group (MRC LMB + Univ. Cambridge,
  Jefferis lab), Google Research (Viren Jain, Michał Januszewski — segmentation).
- Flagship paper: Berg, Beckett, Costa, Schlegel, Januszewski ... Jefferis.
  "Sexual dimorphism in the complete Drosophila male central nervous system connectome". Cell, 3 Sep 2026.
  Preprint: bioRxiv 10.1101/2025.10.09.680999 (v2 Oct 30 2025). Open PMC copy: PMC12636603.
- Companion papers (Cell, Sep 2026): Hoeller et al. "The organization of visual pathways in the Drosophila brain";
  Tastekin et al. taste-feeding connectome; plus a Current Biology paper on sexually dimorphic social-behaviour networks.
- License: CC-BY 4.0.

## Key numbers (from the paper)
| quantity | value |
|---|---|
| neurons (incl. sensory axons) | 166,691 |
| connectome graph | 25.6M edges between 166,391 neurons |
| synapses detected | 46M presynapses -> 312M PSDs (precision/recall 0.82/0.81) |
| "125 million synaptic connections" (press figure) | |
| cell types | 11,691 |
| imaging | eFIB-SEM, 8x8x8 nm isotropic, 0.082 mm^3, 7 machines, 13 months |
| proofreading | ~44 person-years; every fragment >100 synapses reviewed |
| completeness | 94% pre / 42% post synaptic completion in neuropils |
| cross-dataset match | 97.5% of neurons have a type match in FlyWire, hemibrain or MANC |
| fru+ neurons | 4,505 (2,695 high conf) ; dsx+ 407 (332 high conf) |
| sex comparison (central brain, vs FlyWire female) | 7,205 isomorphic types, 114 dimorphic, 262 male-specific, 69 female-specific |

Main biological finding: sensory and motor periphery is nearly identical between sexes; dimorphic and
sex-specific neurons cluster in higher brain centres, forming "circuit switches" that reroute the same
sensory inputs into antagonistic circuits for opposing behaviours (e.g. courtship vs aggression; pC1 split
into 44 types / 19 supertypes).

## Google's part
Flood-filling networks (FFN) for automated segmentation, a refined "PATHFINDER" system trained with
synthetic neurons, automated neuron-type labelling, and Neuroglancer for visualisation.

## Where the data lives
Public GCS bucket, no auth needed: gs://flyem-male-cns  (README at /README_RELEASE_BUCKET.md).
Current release: v1.0 (dirs v0.9, v0.11, v0.13 also present).

- EM image: precomputed://gs://flyem-male-cns/em/em-clahe-jpeg  (shape 94088 x 78317 x 134576, uint8, 8nm)
  raw uncompressed N5: gs://flyem_cns_z0720_07m_dvidcoords_n5
- Segmentation: precomputed://gs://flyem-male-cns/v1.0/segmentation (uint64, 8nm) + meshes + skeletons
  (skeletons-malecns/skeletons-swc/<bodyId>.swc, 8nm units; also mirrored and JRC2018U-template versions)
- Synapses: v1.0/male-cns-v1.0-synapses-precomputed
- ROIs: gs://flyem-male-cns/rois/  (fullbrain-roi-v5 brain neuropils, malecns-vnc-neuropil-roi-v0, optic-lobe columns/layers ...)
- Flat connectome tables (Arrow feather) at v1.0/connectome-data/flat-connectome/:
  | file | size |
  |---|---|
  | body-annotations-male-cns-v1.0-minconf-0.5.feather | 14 MB  (downloaded to data/) |
  | body-neurotransmitters-male-cns-v1.0.feather | 43 MB |
  | body-stats-male-cns-v1.0-minconf-0.5.feather | 778 MB |
  | connectome-weights-...-traced-only.feather | 508 MB (neuron-to-neuron edge list; use this first) |
  | connectome-weights-...-significant-only.feather | 502 MB |
  | connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1.05 GB (all segments) |
  | syn-partners-...(-traced-only).feather | 3.0 GB / 6.8 GB full |
  | syn-points-male-cns-v1.0-minconf-0.5.feather | 13 GB |
  | tbar-neurotransmitters-male-cns-v1.0.feather | 2.7 GB |
- neo4j dump + neuprint input CSVs: v1.0/database/
- Meshes of OTHER datasets transformed into MaleCNS space: flywire2mcns_meshes/783, hemibrain2mcns_meshes/v1.2,
  manc2mcns_meshes/v1.2, banc2mcns_meshes/626
- Neuroglancer scene: https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json

## Query services
- neuPrint: https://neuprint.janelia.org  dataset `male-cns:v1.0` (needs free Google login -> Account -> token).
  Python: `pip install neuprint-python`; `Client('neuprint.janelia.org', dataset='male-cns:v1.0', token=...)`
  R: natverse/malecns (`mcns_neuprint()`, `mcns_neuprint_meta()`, `mcns_connection_table()`, `read_mcns_neurons()`,
  `read_mcns_meshes()`, `mcns_xyz2bodyid()`, `mirror_malecns()`), env var NEUPRINT_TOKEN.
- Clio (annotations): https://clio.janelia.org ; NeuronBridge for light-microscopy matching.
- Cell-type explorer with eyemaps: reiserlab.github.io
- Derived data from the paper: https://github.com/flyconnectome/2025malecns
  (sensorimotor max-flow edges, DN/AN clusters, hSBM communities (311), male-vs-FlyWire edge comparison,
  optic-column assignments, synapse precision/recall by ROI).
- Python ecosystem: navis, navis-flybrains (mirror + template transforms), cloud-volume / tensorstore for volumes.

## Annotation table schema (body-annotations v1.0, 211,577 rows incl. non-neurons)
Columns: bodyId, type, instance, group, superclass, class, subclass, supertype, somaSide, rootSide, status,
statusLabel, flywireType, hemibrainType, mancType, mancBodyid, itoleeHl / trumanHl (hemilineage), birthtime,
fruDsx, dimorphism, entryNerve/exitNerve, receptorType, somaLocation [x,y,z in 8nm voxels], vfbId, synonyms ...
- status == "Traced": 165,122 rows (the neurons). Others: Orphan, Glia, Unimportant, Assign, Anchor.
- superclass counts: ol_intrinsic 89,403 | cb_intrinsic 32,164 | vnc_intrinsic 13,161 | visual_projection 9,201 |
  vnc_sensory 6,370 | ol_sensory 6,098 | cb_sensory 4,868 | ascending 1,846 | descending 1,314 | vnc_motor 708 ...
- fruDsx: fru_high 2,611, fru_low 1,989, dsx_high 138, coexpress_high 193 ...
- dimorphism: male-specific 1,258, sexually dimorphic 771, potentially* 339.
- 11,751 distinct `type` values; 8,199 distinct flywireType matches.

## Sibling datasets
hemibrain (2020, female partial brain, 25k neurons), FlyWire/FAFB (2024, whole female brain, ~140k),
MANC (2023, male VNC), BANC (brain-and-nerve-cord female), optic-lobe (2024, male right OL, a subset of this sample).

## Phase 1 (2026-09-14): the digital fly exists
- flybrain/ package: graph.py (signed CSR from weights + neurotransmitter tables), model.py, checks.py,
  export_activity.py; results in data/graph_brain_summary.json, data/checks.json, data/spectrum.json.
- Weights table schema: body_pre, body_post, weight, type_pre, type_post (25,563,197 rows, 124 M synapses).
  NT table: body, consensus_nt (+ predicted_nt, celltype_predicted_nt, confidences); includes histamine.
- Labellar taste neuron types are LB1a–LB4b, PhG*, claw_tpGRN, dorsal_tpGRN (modality not labelled);
  Shiu 2022 feeding-circuit neurons are named in `synonyms` (G2N-1=GNG232, Roundup=GNG108, Phantom=GNG229,
  Usnea=GNG175, Rattle=GNG132, Zorro=GNG215, Fdg=GNG588, Scapula=GNG087, Bract=DNge173/174, Fudog=DNg67).
- Photoreceptors have no soma in the volume; side inferred from partners: R 2,306 / L 1,737 / unknown 71.
- Leading eigenvalue of the raw signed matrix: 4,089 (excitatory-only 4,767).
