# Floor Plan → 3D Housing Graph Pipeline

Extracts structured topological graphs from floor plan images using **Qwen3-VL-8B-Instruct** (Vision-Language Model), then concatenates 2D graphs into 3D building structures.

**Pipeline at a glance:**

```
  Image ─▶ [1] Classify ─▶ aggregation? ─▶ [2] Extract 2D graph
                         ↘ individual ─▶ skip        │
                                                      ▼
                                         [3] Enrich apartments (room counts)
                                                      │
                                                      ▼
                                         [4] Automated validation
                                                      │
                              needs review? ◀─────────┤
                                   │                  │
                                   ▼                  │
                            [5] RLHF (human hints) ───┤
                                                      │
                                                      ▼
                                         [6] Stack 2D → 3D (N floors)
```

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
│   ├── step1_classify.py                  # Step 1 — classify aggregation/individual
│   ├── step2_aggregation.py               # Step 2 — extract 2D graph
│   ├── step3_apartment_details.py         # Step 3 — enrich apartments
│   ├── step4_validate.py                  # Step 4 — automated validation
│   ├── step5_rlhf.py                      # Step 5 — human feedback loop
│   ├── step6_stack3d.py                   # Step 6 — 2D → 3D stacking
│   ├── pipeline.py                        # End-to-end orchestrator
│   └── utils/
│       ├── json_parser.py
│       └── visualization.py
├── prompts/
│   ├── classify_image.txt
│   ├── analyze_aggregation.txt
│   ├── apartment_details.txt
│   └── rlhf_relearn_connector.txt
├── outputs/
│   ├── graphs/              # 2D + 3D JSON graphs
│   ├── visualizations/      # Side-by-side PNGs (2D) + 3D PNGs
│   ├── logs/                # Classifications, validation reports, human feedback
│   └── stats/               # Aggregate validation summary
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

# After a run, fix the plans that couldn't find stairs/elevators (Step 5)
python -m src.pipeline --rlhf
```

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
| `--rlhf` | Run the interactive Step 5 only |
| `--model-path PATH` | Override the model path |

---

## Pipeline Steps

### Step 1 — Classification (`step1_classify.py`)

Classifies each image as one of:
- **aggregation** — full building floor with multiple apartments around shared corridors/stairs/elevators
- **individual** — single apartment layout or typology catalog

Uses prompt [prompts/classify_image.txt](prompts/classify_image.txt). Only aggregation images proceed. Log saved to `outputs/logs/step1_classification.json`.

### Step 2 — 2D Graph Extraction (`step2_aggregation.py`)

For each aggregation image, the VLM identifies:
1. **Each apartment** (as a node at its principal entrance door)
2. **All staircases** (diagonal parallel lines)
3. **All elevators** (square shafts with X or circle)
4. **Connections** between them via shared corridors/landings

Uses prompt [prompts/analyze_aggregation.txt](prompts/analyze_aggregation.txt). Output: `outputs/graphs/{name}_aggregation.json`.

### Step 3 — Apartment Details (`step3_apartment_details.py`)

For every `apartment` node produced by Step 2, the VLM is asked (one targeted call per apartment, referring to its label + position) to count:

- Number of bedrooms, bathrooms, kitchens, living rooms, terraces
- Presence of an internal corridor
- Total rooms
- Room labels visible inside the apartment

Output is added as a `details` sub-object on each apartment node. Uses prompt [prompts/apartment_details.txt](prompts/apartment_details.txt).

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

### Step 5 — RLHF Human Feedback (`step5_rlhf.py`)

Interactive session for the subset of plans flagged by Step 4 (missing stairs/elevators or unreachable apartments). The CLI:

1. Lists each flagged plan with its current state and reasons.
2. Asks the human for a hint about the connector location (e.g. *"central core between apartments"*, *"labeled ESC on the right"*, *"single-floor house, no connector expected"*).
3. Re-prompts the VLM with the hint baked into [prompts/rlhf_relearn_connector.txt](prompts/rlhf_relearn_connector.txt), which reminds the model what stair/elevator symbols look like.
4. Overwrites the graph with the corrected version (preserving apartment interior details from Step 3).
5. Persists the hint to `outputs/logs/human_feedback_memory.json` so future runs can prepend accumulated feedback.

Trigger with:
```bash
python -m src.pipeline --rlhf
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
  "nodes": [
    {
      "node_id": 0, "type": "apartment", "label": "A", "position": [20, 15],
      "details": {
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
| [prompts/classify_image.txt](prompts/classify_image.txt) | Step 1 | Classify as `aggregation` or `individual` |
| [prompts/analyze_aggregation.txt](prompts/analyze_aggregation.txt) | Step 2 | Extract apartments + stairs/elevators + edges |
| [prompts/apartment_details.txt](prompts/apartment_details.txt) | Step 3 | Count rooms inside a specific apartment |
| [prompts/rlhf_relearn_connector.txt](prompts/rlhf_relearn_connector.txt) | Step 5 | Re-extract graph with human hints about connectors |

---

## Supporting Modules

- **`config.py`** — paths, model config, node-type taxonomy, visualization colors.
- **`vlm_client.py`** — Qwen3-VL-8B wrapper; loads once into GPU memory (bfloat16, ~16 GB VRAM) and exposes `query(image_path, prompt)`.
- **`utils/json_parser.py`** — robust JSON extraction (markdown fences, brace matching, trailing-comma/quote fixes).
- **`utils/visualization.py`** — NetworkX + Matplotlib rendering. `draw_side_by_side()` for 2D, `draw_3d_graph()` for the stacked 3D building.

---

## Hardware Requirements

- **GPU**: NVIDIA with ≥ 16 GB VRAM (RTX 4090, A100, V100 32 GB)
- **RAM**: ≥ 32 GB recommended
- **Disk**: ~16 GB for model weights + room for outputs
