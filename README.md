# Floor Plan → 3D Housing Graph Pipeline

Extracts structured topological graphs from floor plan images using **Qwen3-VL-8B-Instruct** (Vision-Language Model), then concatenates 2D graphs into 3D building structures.

**Pipeline at a glance:**

```
  Image ─▶ [1] Classify ─▶ aggregation? ─▶ [2] Extract 2D graph
                         ↘ individual / other ─▶ skip (logged)  │
                                                                ▼
                                         [3] Enrich apartments (room counts)
                                                      │
                                                      ▼
                                         [4] Automated validation
                                          (reports + flags; does NOT gate)
                                                      │
                                                      ▼
                                         [6] Stack 2D → 3D (N floors)  ── every graph

  Optional side-loop (run manually with --review):
      [4] flags graphs (missing/unreachable connector)
              │
              ▼
      [5] Human-in-the-loop review (hints) ─▶ corrected 2D graph
              │
              ▼
      re-run [4] + [6] via --revalidate
```

Step 1 is **three-class**: `aggregation` / `individual` / `other`. Only `aggregation` plans are extracted; `individual` and `other` (offices, lobbies, parking, pure-circulation floors — no kitchen/bathroom) are logged and skipped.

**Step 4 validates and reports but does not gate the pipeline:** every parseable graph proceeds to Step 6 (3D stacking) regardless of pass/fail. Step 5 (human-in-the-loop) is an *optional, non-blocking* side-loop, run manually with `--review`, that only touches the graphs Step 4 flags for missing/unreachable vertical circulation.

- **Nodes** in the 2D graph: apartments (at their principal entrance door), staircases, elevators
- **Edges** in the 2D graph: physical access between apartments and shared circulation
- **Nodes** in the 3D graph: same, replicated across floors
- **Edges** in the 3D graph: horizontal (within a floor) + vertical (staircase↔staircase, elevator↔elevator between adjacent floors)

---

## Project Structure

```
TFM/
├── JPG/                                   # Input floor plan images
│   ├── IBAVI/   ├── IMPSOL/   └── INCASOL/
├── models/Qwen3-VL-8B-Instruct/           # Downloaded VLM weights
├── src/
│   ├── config.py
│   ├── vlm_client.py
│   ├── step1_classify.py                  # Step 1 — classify aggregation/individual/other
│   ├── step2_aggregation.py               # Step 2 — extract 2D graph
│   ├── step3_apartment_details.py         # Step 3 — enrich apartments
│   ├── step4_validate.py                  # Step 4 — automated validation
│   ├── step5_human_review.py             # Step 5 — human-in-the-loop refinement
│   ├── step6_stack3d.py                   # Step 6 — 2D → 3D stacking
│   ├── pipeline.py                        # End-to-end orchestrator
│   ├── eval_classification.py             # Step 1 accuracy report (sklearn) from labelled CSV
│   ├── eval_spotcheck.py                  # Extraction spot-check summary from labelled CSV
│   ├── stack_custom.py                    # Stack chosen graphs as distinct floors → 3D
│   └── utils/
│       ├── json_parser.py
│       └── visualization.py
├── prompts/
│   ├── classify_image.txt
│   ├── analyze_aggregation.txt
│   ├── apartment_details.txt
│   └── human_review_connector.txt
├── outputs/
│   ├── graphs/                  # 2D + 3D JSON graphs (with captured reasoning fields)
│   ├── visualizations/          # Side-by-side PNGs (2D) + 3D PNGs
│   ├── human_review_examples/   # Persistent stair/elevator crops (Step 5 few-shot memory)
│   ├── logs/                    # Classifications, validation reports, human feedback
│   └── stats/                   # Aggregate validation summary
├── download_model.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
pip install huggingface_hub qwen-vl-utils
python download_model.py                   # downloads ~16 GB model
```

## Run

```bash
# Full pipeline on all datasets (Steps 1-4 + 6)
python -m src.pipeline

# Quick test on 5 images from IBAVI
python -m src.pipeline --dataset IBAVI --limit 5

# After a run, fix the plans that couldn't find stairs/elevators (Step 5, interactive)
python -m src.pipeline --review

# After review, refresh validation + 3D from the corrected graphs (Steps 4+6, no VLM)
python -m src.pipeline --revalidate
```

**Typical end-to-end workflow** (three mutually-exclusive modes):

1. `python -m src.pipeline` — automatic batch: Steps 1–4 + 6. Step 4 writes the list of plans that need review.
2. `python -m src.pipeline --review` — interactive Step 5: walk the flagged plans, give hints, overwrite the corrected graphs.
3. `python -m src.pipeline --revalidate` — re-run Steps 4 + 6 against the corrected graphs so the validation summary and 3D graphs reflect the fixes (no model load — fast).

**CLI flags:**

| Flag | Description |
|------|-------------|
| `--dataset IBAVI\|IMPSOL\|INCASOL` | Process only one dataset |
| `--limit N` | Process at most N images (for testing) |
| `--no-vis` | Skip PNG visualization generation |
| `--no-enrich` | Skip Step 3 (apartment detail extraction) |
| `--no-validate` | Skip Step 4 (automated validation) |
| `--no-stack` | Skip Step 6 (3D stacking) |
| `--n-floors N` | Number of floors for 3D stacking (default: 3) |
| `--review` | Run the interactive Step 5 only (human-in-the-loop refinement). Legacy alias: `--rlhf` |
| `--revalidate` | Re-run only Steps 4+6 on the graphs already in `outputs/graphs/` (no VLM). Use after `--review` |
| `--model-path PATH` | Override the model path |

---

## Pipeline Steps

### Step 1 — Classification (`step1_classify.py`)

Classifies each image as one of three classes:
- **aggregation** — full building floor with multiple apartments around shared corridors/stairs/elevators
- **individual** — single apartment layout or typology catalog
- **other** — not a habitable dwelling at all: office, retail/commercial locale, lobby, parking, technical/storage, or pure-circulation floor (no kitchen AND no bathroom)

Deterministic greedy decoding (`do_sample=False`) for a stable one-word answer. Two few-shot reference plans (one `aggregation`, one `individual`) are prepended to anchor the decision. The decision procedure leads with kitchen/bathroom counting (zero of both → `other`), so a stair core alone never forces an "aggregation".

Uses prompt [prompts/classify_image.txt](prompts/classify_image.txt). **Only `aggregation` images proceed to Steps 2–3**; `individual`, `other`, and unparseable (`unknown`) results are logged and skipped. Log saved to `outputs/logs/step1_classification.json`.

### Step 2 — 2D Graph Extraction (`step2_aggregation.py`)

For each aggregation image, the VLM identifies:
1. **Each apartment** (as a node at its principal entrance door)
2. **All staircases** (diagonal parallel lines)
3. **All elevators** (square shafts with X or circle)
4. **Connections** between them via shared corridors/landings (clean hub-and-spoke: apartments link to connectors, not to each other)

Two key behaviors:
- **Captured reasoning** — the prompt asks the model to emit a `building_analysis` field *inside* the JSON (instructed chain-of-thought, kept machine-parseable) describing the floor's spatial arrangement *before* the coordinates. This trace is saved at the top of the graph JSON for the manual spot-check.
- **No-fabrication edge rule** — not every apartment must reach a connector. The model is told to emit only edges it can actually trace; a missing edge beats an invented one. Genuinely under-connected graphs are caught by Step 4 and routed to Step 5.

Uses prompt [prompts/analyze_aggregation.txt](prompts/analyze_aggregation.txt). Output: `outputs/graphs/{name}_aggregation.json`.

### Step 3 — Apartment Details (`step3_apartment_details.py`)

For every `apartment` node produced by Step 2, the VLM is asked (one targeted call per apartment, referring to its label + position) to count:

- Number of bedrooms, bathrooms, kitchens, living rooms, terraces
- Presence of an internal corridor
- Total rooms
- Room labels visible inside the apartment

The prompt also asks for a `spatial_analysis` reasoning field (where the apartment sits, which walls bound it) so neighboring rooms aren't miscounted; this trace is preserved per apartment. Schema examples are **concrete typed values** (`true`/`false`/`int`), not string descriptions — avoiding the `bool("...") == True` trap on boolean fields like `has_corridor`.

Output is added as a `details` sub-object on each apartment node (including `spatial_analysis`). Uses prompt [prompts/apartment_details.txt](prompts/apartment_details.txt).

### Step 4 — Automated Validation (`step4_validate.py`)

Runs six checks per graph:

| Check | Meaning |
|-------|---------|
| `has_apartments` | ≥ 2 apartments present (hard) |
| `has_connectors` | ≥ 1 staircase or elevator present (soft) |
| `all_connected` | single connected component (hard) |
| `no_isolated_apt` | every apartment has at least one edge (hard) |
| `every_apt_reaches_connector` | every apartment can reach a stair/elevator via graph (soft) |
| `no_dup_positions` | no two nodes stacked at identical [x, y] |

Each graph gets a `passed` flag, a `needs_human_review` flag (true when connectors are missing or unreachable), and a `score` (fraction of checks passed). Results: `outputs/stats/validation_summary.json`, `outputs/logs/validation_reports.json`, `outputs/logs/needs_human_review.json`.

### Step 5 — Human-in-the-Loop Refinement (`step5_human_review.py`)

Interactive session for the subset of plans flagged by Step 4 (missing stairs/elevators or unreachable apartments). For each flagged plan the CLI:

1. Lists the plan with its current state and the reasons it was flagged.
2. Asks the human for a **text hint** about the connector location (e.g. *"central core between apartments"*, *"staircase is the parallel-line block at lower-right"*, *"single-floor house, no connector expected"*). **Press ENTER to skip a plan.**
3. *(Optional)* Asks for a **reference image** — a small **crop of the staircase/elevator symbol itself** (not a full plan). If provided, it is copied into `outputs/human_review_examples/{staircase,elevator}/` and reused as a **persistent few-shot visual example on every future review call**, this session and later ones. **Press ENTER to skip** — the text hint alone is enough.
4. Re-prompts the VLM with the hint (+ any stored crops) using [prompts/human_review_connector.txt](prompts/human_review_connector.txt), which describes the stair/elevator symbols and adds a `building_analysis` reasoning field.
5. Overwrites the graph with the corrected version (preserving apartment interior details from Step 3). If the model confirms no connector exists, it records `no_connector_confirmed: true` instead of inventing one.
6. Persists the hint to `outputs/logs/human_feedback_memory.json` so future runs can prepend accumulated feedback.

Step 5 only rewrites the 2D graphs — it does **not** re-run Steps 4/6. After review, run `--revalidate` to refresh the validation summary and 3D graphs from the corrected graphs.

Trigger with:
```bash
python -m src.pipeline --review        # interactive
python -m src.pipeline --revalidate    # then refresh Steps 4+6 (no VLM)
```

### Step 6 — 2D → 3D Stacking (`step6_stack3d.py`)

Replicates each 2D graph N times (default 3) to simulate an N-floor building and connects adjacent floors via their staircase and elevator nodes.

Node IDs are globalized as `floor * 1_000_000 + original_id` so (floor, original_id) is recoverable. Edges are tagged `horizontal` (within a floor) or `vertical` (between floors via a connector). Connectors on adjacent floors are matched by nearest position.

Output: `outputs/graphs/{name}_3d.json`, plus a 3D scatter-plot visualization in `outputs/visualizations/{name}_3d.png`.

---

## Output Formats

**2D aggregation graph** (`*_aggregation.json`):

```json
{
  "source_file": "inca_05.jpg",
  "dataset": "INCASOL",
  "type": "aggregation",
  "building_analysis": "Central stair-and-elevator core serving 4 perimeter apartments.",
  "nodes": [
    {
      "node_id": 0, "type": "apartment", "label": "A", "position": [20, 15],
      "details": {
        "spatial_analysis": "Apartment A occupies the upper-left quadrant, bounded by the central corridor to the south.",
        "num_bedrooms": 2, "num_bathrooms": 1,
        "num_living_rooms": 1, "num_kitchens": 1, "num_terraces": 1,
        "has_corridor": true, "total_rooms": 6,
        "room_labels": ["H1", "H2", "EMC", "B", "P", "Te"]
      }
    },
    {"node_id": 4, "type": "staircase", "label": null, "position": [50, 50]},
    {"node_id": 5, "type": "elevator",  "label": null, "position": [52, 45]}
  ],
  "edges": [
    {"from_node_id": 0, "to_node_id": 4},
    {"from_node_id": 4, "to_node_id": 5}
  ],
  "metadata": {
    "num_apartments": 4, "num_staircases": 1, "num_elevators": 1,
    "num_edges": 5, "apartments_with_details": 4
  },
  "flags": []
}
```

**3D building graph** (`*_3d.json`):

```json
{
  "source_file": "inca_05.jpg",
  "dataset": "INCASOL",
  "type": "building_3d",
  "n_floors": 3,
  "nodes": [
    {"node_id": 4, "floor": 0, "type": "staircase", "position": [50, 50]},
    {"node_id": 1000004, "floor": 1, "type": "staircase", "position": [50, 50]},
    {"node_id": 2000004, "floor": 2, "type": "staircase", "position": [50, 50]}
  ],
  "edges": [
    {"from_node_id": 0, "to_node_id": 4, "edge_type": "horizontal", "floor": 0},
    {"from_node_id": 4, "to_node_id": 1000004, "edge_type": "vertical", "via": "staircase"},
    {"from_node_id": 1000004, "to_node_id": 2000004, "edge_type": "vertical", "via": "staircase"}
  ],
  "metadata": {
    "num_apartments": 12, "num_staircases": 3, "num_elevators": 3,
    "num_horizontal_edges": 15, "num_vertical_edges": 4
  }
}
```

---

## Prompt Templates

| File | Used in | Purpose |
|------|---------|---------|
| [prompts/classify_image.txt](prompts/classify_image.txt) | Step 1 | Classify as `aggregation` / `individual` / `other` (one word) |
| [prompts/analyze_aggregation.txt](prompts/analyze_aggregation.txt) | Step 2 | Extract apartments + stairs/elevators + edges; `building_analysis` reasoning; no-fabrication edge rule |
| [prompts/apartment_details.txt](prompts/apartment_details.txt) | Step 3 | Count rooms inside a specific apartment; `spatial_analysis` reasoning; typed schema examples |
| [prompts/human_review_connector.txt](prompts/human_review_connector.txt) | Step 5 | Re-extract graph with human hints about connectors; `building_analysis` reasoning |

---

## Supporting Modules

- **`config.py`** — paths, model config, node-type taxonomy, visualization colors.
- **`vlm_client.py`** — Qwen3-VL-8B wrapper; loads once into GPU memory (bfloat16, ~16 GB VRAM) and exposes `query(image_path, prompt)`.
- **`utils/json_parser.py`** — robust JSON extraction (markdown fences, brace matching, trailing-comma/quote fixes).
- **`utils/visualization.py`** — NetworkX + Matplotlib rendering. `draw_side_by_side()` for 2D, `draw_3d_graph()` for the stacked 3D building (translucent floor planes, even floor spacing).

---

## Analysis Scripts

- **`eval_classification.py`** — Step 1 accuracy evaluation. Reads `outputs/stats/classification_eval.csv` (columns `file,dataset,predicted,truth`; fill `truth` by hand) and prints scikit-learn's `classification_report` (per-class precision/recall/F1), overall accuracy, and a confusion matrix. `--save` also writes `outputs/stats/classification_report.txt`.
  ```bash
  python -m src.eval_classification --save
  ```
- **`eval_spotcheck.py`** — extraction spot-check summary. Reads `outputs/stats/spotcheck_sample.csv` (manually filled: `missing_nodes`, `spurious_nodes`, `wrong_edges`, `wrong_rooms`, `fully_correct`, plus `true_class`) and reports the fully-correct rate and per-category error rates over the true-aggregation graphs, separating out misclassified (leakage) plans. `--save` writes `outputs/stats/spotcheck_report.txt`.
  ```bash
  python -m src.eval_spotcheck --save
  ```
- **`stack_custom.py`** — build one 3D building from several *different* 2D graphs (one per floor, bottom-up), instead of replicating a single floor. Connectors on adjacent floors are linked by nearest position. Works best when the floors share a coordinate system (0–100%).
  ```bash
  python -m src.stack_custom ibavi_34 ibavi_35 --out ibavi_block
  ```

---

## Hardware Requirements

- **GPU**: NVIDIA with ≥ 16 GB VRAM (RTX 4090, A100, V100 32 GB)
- **RAM**: ≥ 32 GB recommended
- **Disk**: ~16 GB for model weights + room for outputs
