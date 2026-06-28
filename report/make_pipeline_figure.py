"""
Generate the six-stage pipeline diagram for the 'Proposed Pipeline' section
(Chapter 3).

Flow: Image -> Step 1 (classify) -> [aggregation] -> Steps 2-3 (VLM extract)
-> Step 4 (validate) -> Step 6 (3D stacking). EVERY extracted graph flows
Step 4 -> Step 6 unconditionally; Step 4 only validates and reports, it does
not gate Step 6.

Step 5 (human-in-the-loop) is an OPTIONAL, non-blocking side-loop, run manually
with --review: graphs flagged by the diagnostic (soft) checks can be corrected
in Step 5, the corrected 2D graph replaces the original and is re-validated, but
the main Step 4 -> Step 6 flow is unaffected. Drawn dashed to mark it optional.
individual / other plans are skipped at Step 1.

Run from the report/ directory:
    python make_pipeline_figure.py
Produces: figures/pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(9, 11.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 14.5)
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


def diamond(x, y, w, h, text, color, text_color="white", fontsize=9):
    """Decision node drawn as a diamond."""
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=color,
                             edgecolor="black", linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight="bold")


def arrow(x1, y1, x2, y2, text="", color="black", lbldx=0.25, lbldy=0.0,
          lblside="left", connectionstyle=None, fontsize=8, linestyle="-"):
    kw = dict(arrowstyle="-|>", mutation_scale=16, color=color,
              linewidth=1.7, shrinkA=2, shrinkB=2, linestyle=linestyle)
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))
    if text:
        ha = "left" if lblside == "left" else "right"
        ax.text((x1 + x2) / 2 + lbldx, (y1 + y2) / 2 + lbldy, text, ha=ha,
                va="center", fontsize=fontsize, style="italic", color=color)


cx = 3.6   # main column
rx = 8.3   # right column (Step 5 side-branch + skip)

# --- Main vertical spine ---
box(cx, 13.6, 3.0, 0.8, "Floor-plan image", NEUTRAL)
box(cx, 12.1, 3.7, 0.9, "Step 1\nClassification (VLM)", VLM)
box(cx, 10.5, 3.7, 1.0, "Step 2\nAggregation extraction (VLM)", VLM)
box(cx,  8.9, 3.7, 1.0, "Step 3\nApartment detailing (VLM)", VLM)
box(cx,  6.8, 4.1, 1.0, "Step 4\nStructural validation\n& routing (no VLM)", DET,
    fontsize=9)
box(cx,  3.7, 3.7, 0.9, "Step 6\n3D stacking (no VLM)", DET)
box(cx,  2.1, 3.2, 0.8, "2D + 3D graph JSON", NEUTRAL)

# --- Step 5 as an OPTIONAL, non-blocking side-loop off Step 4 ---
box(rx, 6.8, 3.0, 1.3, "Step 5\nHuman-in-the-loop\nrefinement (optional,\n--review)",
    HITL, fontsize=8.5)

# --- Skip box (individual / other) off Step 1 ---
box(rx, 12.1, 2.6, 1.0, "individual / other\n→ logged, skipped", SKIP,
    text_color="black", fontsize=9)

# --- Main-flow arrows (top spine): EVERY graph flows Step 4 -> Step 6 ---
arrow(cx, 13.2, cx, 12.55)
arrow(cx, 11.65, cx, 11.0, "aggregation")
arrow(cx, 10.0, cx, 9.4)
arrow(cx, 8.4, cx, 7.75)
arrow(cx, 6.3, cx, 4.15, "all graphs", lbldx=0.25, lblside="left", color=DET)
arrow(cx, 3.25, cx, 2.5)                # Step 6 -> output JSON

# --- Step 1 -> skip ---
arrow(cx + 1.85, 12.1, rx - 1.3, 12.1)

# --- OPTIONAL review loop (dashed): flagged graphs detour Step 4 -> Step 5 ... ---
arrow(cx + 2.05, 7.05, rx - 1.5, 7.05,
      "flagged by\ndiagnostic checks", color=HITL,
      lbldx=0.0, lbldy=0.55, lblside="left", fontsize=7, linestyle=(0, (5, 3)))

# --- ... Step 5 corrected -> back to Step 4; main flow to Step 6 is unaffected ---
arrow(rx - 1.5, 6.55, cx + 2.05, 6.55,
      "corrected graph\nreplaces 2D, re-validated", color=HITL,
      lblside="right", lbldx=0.0, lbldy=-0.55, fontsize=7, linestyle=(0, (5, 3)))

# --- Legend ---
legend = [
    Patch(facecolor=VLM, edgecolor="black", label="VLM-driven step"),
    Patch(facecolor=DET, edgecolor="black", label="Deterministic step (no VLM)"),
    Patch(facecolor=HITL, edgecolor="black", label="Human-in-the-loop (optional review loop)"),
    Patch(facecolor=SKIP, edgecolor="black", label="Skipped from extraction"),
]
ax.legend(handles=legend, loc="lower center", ncol=2, fontsize=9,
          frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout()
out = "figures/pipeline.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
