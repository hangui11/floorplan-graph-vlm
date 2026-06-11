"""
Step 5: Reinforcement Learning with Human Feedback (RLHF-style).

For graphs flagged as needing human review (typically: missing staircase
or elevator — no connection component between apartments), this step:

1. Lists the flagged plans to the user.
2. Accepts human-provided hints about where/what the connector looks like.
3. Sends a re-learning prompt to the VLM with the hints baked in.
4. Replaces the graph with the corrected version.

This is a lightweight RLHF-style loop: the "reward signal" is the human
hint, and the model adapts its output in a single corrective step.
For persistent learning across runs, the hints are stored in a feedback
memory file and prepended to all future re-learn prompts.
"""
import json
import logging
import shutil
from pathlib import Path

from src.config import PROMPTS_DIR, GRAPHS_DIR, LOGS_DIR, OUTPUT_DIR
from src.vlm_client import VLMClient
from src.step2_aggregation import _normalize_nodes, _normalize_edges, build_aggregation_graph
from src.step4_validate import validate_graph
from src.utils.json_parser import extract_json

logger = logging.getLogger(__name__)

FEEDBACK_MEMORY_FILE = LOGS_DIR / "human_feedback_memory.json"
EXAMPLES_DIR = OUTPUT_DIR / "rlhf_examples"
MAX_EXAMPLES_PER_TYPE = 3  # cap to keep VLM context manageable


# ── Feedback memory (persistent) ───────────────────────────────────────────────

def load_feedback_memory() -> list[dict]:
    """Load accumulated human hints from prior sessions."""
    if FEEDBACK_MEMORY_FILE.exists():
        with open(FEEDBACK_MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_feedback_memory(memory: list[dict]):
    """Persist human hints for future runs."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_feedback(source_file: str, hint: str, memory: list[dict] | None = None) -> list[dict]:
    """Append a new human hint and persist."""
    memory = memory if memory is not None else load_feedback_memory()
    memory.append({"source_file": source_file, "hint": hint})
    save_feedback_memory(memory)
    return memory


# ── Visual example memory (persistent) ────────────────────────────────────────

def save_example_image(src_path: Path, element_type: str) -> Path | None:
    """
    Copy a user-provided reference image into outputs/rlhf_examples/{type}/
    so it is available as few-shot context for every future re-learn call.

    Args:
        src_path: Path to the user's reference image (a crop or full image).
        element_type: "staircase" or "elevator".

    Returns:
        Path to the saved example, or None if the source does not exist.
    """
    if element_type not in ("staircase", "elevator"):
        raise ValueError(f"element_type must be 'staircase' or 'elevator', got {element_type}")

    src_path = Path(src_path)
    if not src_path.exists():
        logger.warning(f"Example image not found: {src_path}")
        return None

    target_dir = EXAMPLES_DIR / element_type
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(target_dir.glob("*"))
    idx = len(existing) + 1
    target = target_dir / f"{element_type}_{idx:03d}{src_path.suffix.lower()}"
    shutil.copy(src_path, target)
    logger.info(f"Saved {element_type} example to {target}")
    return target


def load_examples(max_per_type: int = MAX_EXAMPLES_PER_TYPE) -> list[tuple[Path, str]]:
    """
    Load the most recent few-shot examples for stairs and elevators.

    Returns a list of (image_path, caption) tuples suitable for VLMClient.query().
    Newest examples are preferred (later files).
    """
    examples: list[tuple[Path, str]] = []
    for element_type, caption in (
        ("staircase", "Reference image: this is a STAIRCASE as it appears in these floor plans."),
        ("elevator",  "Reference image: this is an ELEVATOR as it appears in these floor plans."),
    ):
        type_dir = EXAMPLES_DIR / element_type
        if not type_dir.exists():
            continue
        files = sorted(type_dir.glob("*"))[-max_per_type:]
        for f in files:
            examples.append((f, caption))
    return examples


# ── Re-learn with feedback ─────────────────────────────────────────────────────

def load_relearn_prompt() -> str:
    return (PROMPTS_DIR / "rlhf_relearn_connector.txt").read_text(encoding="utf-8")


def relearn_graph(
    client: VLMClient,
    image_path: Path,
    graph: dict,
    hints: list[str],
    use_examples: bool = True,
) -> dict | None:
    """
    Re-prompt the VLM with human hints + stored visual examples and return
    a fresh aggregation graph.

    Args:
        client: Loaded VLMClient.
        image_path: Path to the aggregation floor plan.
        graph: The current (flagged) graph.
        hints: List of human-written hint strings.
        use_examples: If True, prepend accumulated stair/elevator examples
            from outputs/rlhf_examples/ as few-shot context.

    Returns:
        A new graph dict (replacement), or None on failure.
    """
    prompt_template = load_relearn_prompt()
    hints_text = "\n".join(f"- {h}" for h in hints) if hints else "- (no hints provided)"
    prompt = prompt_template.replace("{human_hints}", hints_text)

    example_images = load_examples() if use_examples else []
    if example_images:
        logger.info(f"  Using {len(example_images)} reference example(s) as few-shot context.")

    response = client.query(image_path, prompt, example_images=example_images)
    result = extract_json(response)

    if result is None:
        logger.error(f"Re-learn failed: could not parse JSON for {image_path.name}")
        return None

    if "nodes" not in result or "edges" not in result:
        logger.error(f"Re-learn failed: missing nodes/edges for {image_path.name}")
        return None

    # Normalize the new output
    analysis = {
        "nodes": _normalize_nodes(result["nodes"]),
        "edges": [],
    }
    analysis["edges"] = _normalize_edges(result["edges"], analysis["nodes"])

    new_graph = build_aggregation_graph(
        source_file=graph["source_file"],
        dataset=graph.get("dataset", "unknown"),
        analysis=analysis,
    )

    # Preserve apartment interior details from the previous graph (by label)
    old_details_by_label = {
        n.get("label"): n.get("details")
        for n in graph["nodes"]
        if n["type"] == "apartment" and n.get("details")
    }
    for node in new_graph["nodes"]:
        if node["type"] == "apartment" and node.get("label") in old_details_by_label:
            node["details"] = old_details_by_label[node["label"]]

    # Record that this graph went through RLHF
    new_graph["rlhf"] = {
        "applied": True,
        "hints_used": hints,
        "no_connector_confirmed": bool(result.get("no_connector_confirmed", False)),
    }

    return new_graph


# ── Reporting helpers ──────────────────────────────────────────────────────────

def list_graphs_needing_review(graphs_dir: Path | None = None) -> list[dict]:
    """
    Return the list of graphs that need human review, with summary info.
    """
    needs_review_file = LOGS_DIR / "needs_human_review.json"
    if not needs_review_file.exists():
        logger.warning(
            f"{needs_review_file} not found. Run Step 4 validation first."
        )
        return []

    with open(needs_review_file, "r", encoding="utf-8") as f:
        sources = json.load(f)

    gdir = graphs_dir or GRAPHS_DIR
    results = []
    for source in sources:
        stem = Path(source).stem
        graph_path = gdir / f"{stem}_aggregation.json"
        if not graph_path.exists():
            continue
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        report = validate_graph(graph)
        reasons = []
        if not report["checks"]["has_connectors"]["passed"]:
            reasons.append("no staircase or elevator detected")
        if not report["checks"]["every_apt_reaches_connector"]["passed"]:
            reasons.append(
                f"{len(report['checks']['every_apt_reaches_connector']['unreachable'])} apartment(s) cannot reach a connector"
            )

        results.append({
            "source_file": source,
            "dataset": graph.get("dataset"),
            "graph_json": str(graph_path),
            "num_apartments": graph["metadata"].get("num_apartments", 0),
            "num_staircases": graph["metadata"].get("num_staircases", 0),
            "num_elevators": graph["metadata"].get("num_elevators", 0),
            "reasons": reasons,
        })

    return results


def print_review_report(items: list[dict]):
    """Pretty-print a review report for the user."""
    if not items:
        print("\n  No graphs need human review.\n")
        return

    print(f"\n  {'='*70}")
    print(f"  {len(items)} plan(s) need human review (missing connectors):")
    print(f"  {'='*70}")
    for i, it in enumerate(items, 1):
        print(f"\n  [{i}] {it['source_file']}  ({it['dataset']})")
        print(f"       apartments={it['num_apartments']}, "
              f"staircases={it['num_staircases']}, "
              f"elevators={it['num_elevators']}")
        for r in it["reasons"]:
            print(f"       - {r}")
    print(f"\n  {'='*70}\n")


# ── Interactive session ────────────────────────────────────────────────────────

def run_interactive_rlhf(
    client: VLMClient,
    visualize: bool = True,
):
    """
    Interactive command-line session: walks the user through each flagged graph,
    collects hints, and re-prompts the VLM.
    """
    items = list_graphs_needing_review()
    print_review_report(items)
    if not items:
        return

    memory = load_feedback_memory()

    for it in items:
        source_file = it["source_file"]
        graph_path = Path(it["graph_json"])

        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        image_path = Path(source_file)

        print(f"\n--- {source_file} ---")
        print(f"Current state: {it['num_apartments']} apartments, "
              f"{it['num_staircases']} stairs, {it['num_elevators']} elevators.")

        # Show currently-remembered visual examples
        current_examples = load_examples()
        if current_examples:
            print(f"Remembered visual examples: {len(current_examples)} "
                  f"(stair/elevator crops from past sessions)")

        print("\nProvide a hint about where the staircase/elevator is located")
        print("(e.g. 'central core between apartments', 'right side labeled E',")
        print("'there is no connector — this is a single-floor house').")
        print("Leave empty to skip this plan.")
        hint = input("Hint: ").strip()

        if not hint:
            print("  Skipped.")
            continue

        # Optional: teach the VLM by example for this and future sessions
        print("\nOptional — provide reference image(s) so the VLM learns what")
        print("a staircase/elevator looks like in these plans. These are reused")
        print("in every future RLHF call. Leave empty to skip.")
        while True:
            ex_path = input("  Example image path (or ENTER to continue): ").strip().strip('"')
            if not ex_path:
                break
            ex_type = input("  Type [staircase/elevator]: ").strip().lower()
            if ex_type not in ("staircase", "elevator"):
                print("    Must be 'staircase' or 'elevator'. Skipped.")
                continue
            saved = save_example_image(Path(ex_path), ex_type)
            if saved:
                print(f"    Stored: {saved}")

        # Persist the hint
        memory = add_feedback(source_file, hint, memory)

        # Re-learn
        print("  Re-learning with hint...")
        new_graph = relearn_graph(client, image_path, graph, [hint])
        if new_graph is None:
            print("  Re-learn FAILED.")
            continue

        # Save the updated graph
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(new_graph, f, indent=2, ensure_ascii=False)

        meta = new_graph["metadata"]
        print(
            f"  Updated: {meta['num_apartments']} apartments, "
            f"{meta['num_staircases']} stairs, {meta['num_elevators']} elevators."
        )

        if visualize:
            try:
                from src.config import VIS_DIR
                from src.utils.visualization import draw_side_by_side
                VIS_DIR.mkdir(parents=True, exist_ok=True)
                vis_path = VIS_DIR / f"{image_path.stem}_aggregation_rlhf.png"
                draw_side_by_side(
                    image_path, new_graph,
                    save_path=vis_path,
                    title=f"{new_graph['dataset']} / {image_path.name} (after RLHF)",
                )
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

    print("\nRLHF session complete.")
