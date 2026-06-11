"""
Configuration for the floor plan to housing graph pipeline.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JPG_DIR = PROJECT_ROOT / "JPG"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
VIS_DIR = OUTPUT_DIR / "visualizations"
LOGS_DIR = OUTPUT_DIR / "logs"
STATS_DIR = OUTPUT_DIR / "stats"

DATASETS = {
    "IBAVI": JPG_DIR / "IBAVI",
    "IMPSOL": JPG_DIR / "IMPSOL",
    "INCASOL": JPG_DIR / "INCASOL",
}

# ── Model ──────────────────────────────────────────────────────────────────────
# MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_LOCAL_PATH = str(PROJECT_ROOT / "models" / "Qwen3-VL-4B-Instruct")
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.1
TOP_P = 0.9

# ── Image classification labels ───────────────────────────────────────────────
IMAGE_TYPES = ["aggregation", "individual"]

# ── Aggregation node types ─────────────────────────────────────────────────────
# In an aggregation block, each apartment is a node. These are the special
# circulation/vertical elements that also become nodes.
VERTICAL_ELEMENT_TYPES = ["staircase", "elevator"]

# ── One-shot examples (canonical reference plans) ──────────────────────────────
EXAMPLE_AGGREGATION_IMG = JPG_DIR / "INCASOL" / "inca_05.jpg"
EXAMPLE_INDIVIDUAL_IMG  = JPG_DIR / "INCASOL" / "inca_50.jpg"

# ── Visualization ──────────────────────────────────────────────────────────────
NODE_COLORS = {
    "apartment":   "#4A90D9",
    "staircase":   "#E74C3C",
    "elevator":    "#F39C12",
    "corridor":    "#9B9B9B",
    "courtyard":   "#27AE60",
}
