"""
Generate the six-stage pipeline diagram for the 'Proposed Pipeline' section
(Chapter 3).

Flow: Image -> Step 1 (classify) -> [aggregation] -> Steps 2-3 (VLM extract)
-> Step 4 (validate). Step 4 BRANCHES: graphs that pass go straight to Step 6;
graphs flagged for review take an optional detour to Step 5 (interactive HITL)
and are then re-validated. individual/other plans are skipped at Step 1.

Run from the report/ directory:
    python make_pipeline_figure.py
Produces: figures/pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 11)
ax.set_ylim(0, 14)
ax.axis("off")

VLM = "#4A90D9"      # VLM-driven steps
DET = "#27AE60"      # deterministic (no VLM) steps
HITL = "#F39C12"     # human-in-the-loop (optional)
SKIP = "#BDC3C7"     # skipped
NEUTRAL = "#34495E"

def box(x, y, w, h, text, color, text_color="white", fontsize=10):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=color, edgecolor="black", linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight="bold")

def arrow(x1, y1, x2, y2, text="", color="black", lbldx=0.25, lblside="left"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        color=color, linewidth=1.6, shrinkA=2, shrinkB=2))
    if text:
        ha = "left" if lblside == "left" else "right"
        ax.text((x1 + x2) / 2 + lbldx, (y1 + y2) / 2, text, ha=ha,
                va="center", fontsize=8, style="italic", color=color)

cx = 3.7   # main column
rx = 8.2   # right column (Step 5 side-branch)

# Main vertical spine
box(cx, 13.3, 3.0, 0.8, "Floor-plan image", NEUTRAL)
box(cx, 11.9, 3.6, 0.9, "Step 1\nClassification (VLM)", VLM)
box(cx, 10.3, 3.6, 1.0, "Step 2\nAggregation extraction (VLM)", VLM)
box(cx,  8.7, 3.6, 1.0, "Step 3\nApartment detailing (VLM)", VLM)
box(cx,  7.1, 3.6, 0.9, "Step 4\nValidation (no VLM)", DET)
box(cx,  3.9, 3.6, 0.9, "Step 6\n3D stacking (no VLM)", DET)
box(cx,  2.4, 3.2, 0.8, "2D + 3D graph JSON", NEUTRAL)

# Step 5 as an optional side-branch off Step 4
box(rx, 5.5, 3.0, 1.0, "Step 5\nHuman-in-the-loop\nreview (optional)", HITL, fontsize=9)

# Skip box (individual / other) off Step 1
box(rx, 11.9, 2.6, 1.0, "individual / other\n→ logged, skipped", SKIP,
    text_color="black", fontsize=9)

# Main-flow arrows
arrow(cx, 12.9, cx, 12.35)
arrow(cx, 11.45, cx, 10.8, "aggregation")
arrow(cx, 9.8, cx, 9.2)
arrow(cx, 8.2, cx, 7.55)
arrow(cx, 6.65, cx, 4.35, "passes", lblside="left")   # Step 4 -> Step 6 (most graphs)
arrow(cx, 3.45, cx, 2.8)

# Step 1 -> skip
arrow(cx + 1.8, 11.9, rx - 1.3, 11.9)

# Step 4 -> Step 5 (flagged detour) and Step 5 -> back to Step 4 (re-validate)
arrow(cx + 1.8, 7.1, rx - 1.5, 6.0, "needs\nreview", color=HITL, lbldx=-0.1)
arrow(rx - 1.5, 5.1, cx + 1.8, 6.7, "corrected,\nre-validate", color=HITL, lblside="right", lbldx=-0.2)

# Legend
legend = [
    Patch(facecolor=VLM, edgecolor="black", label="VLM-driven step"),
    Patch(facecolor=DET, edgecolor="black", label="Deterministic step (no VLM)"),
    Patch(facecolor=HITL, edgecolor="black", label="Human-in-the-loop (optional)"),
    Patch(facecolor=SKIP, edgecolor="black", label="Skipped from extraction"),
]
ax.legend(handles=legend, loc="lower center", ncol=2, fontsize=9,
          frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout()
out = "figures/pipeline.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
