"""
Step 2: Analyze aggregation floor plans.

For each aggregation image, detect:
- Individual apartments as nodes (located at their principal entrance door)
- Staircases and elevators as nodes
- Connections between them (which apartments share access to which stairs/elevators)
"""
import json
import logging
from pathlib import Path

import networkx as nx

from src.config import (
    PROMPTS_DIR,
    GRAPHS_DIR,
    VERTICAL_ELEMENT_TYPES,
    EXAMPLE_AGGREGATION_IMG,
)
from src.vlm_client import VLMClient
from src.utils.json_parser import extract_json

logger = logging.getLogger(__name__)


# One-shot reference: canonical aggregation plan (inca_05.jpg) paired with
# its expected JSON output. Teaches the model the visual pattern and the
# exact response schema. Skipped automatically if the target is this image.
_EXAMPLE_AGGREGATION_JSON = """{
  "nodes": [
    {"node_id": 0, "type": "apartment", "label": "A", "door_position": [30, 22]},
    {"node_id": 1, "type": "apartment", "label": "A", "door_position": [70, 22]},
    {"node_id": 2, "type": "apartment", "label": "B", "door_position": [30, 50]},
    {"node_id": 3, "type": "apartment", "label": "B", "door_position": [70, 50]},
    {"node_id": 4, "type": "apartment", "label": "A", "door_position": [30, 78]},
    {"node_id": 5, "type": "apartment", "label": "A", "door_position": [70, 78]},
    {"node_id": 6, "type": "staircase", "label": null, "position": [50, 50]},
    {"node_id": 7, "type": "elevator",  "label": null, "position": [40, 50]},
    {"node_id": 8, "type": "elevator",  "label": null, "position": [60, 50]}
  ],
  "edges": [
    {"from_node_id": 0, "to_node_id": 6},
    {"from_node_id": 1, "to_node_id": 6},
    {"from_node_id": 2, "to_node_id": 6},
    {"from_node_id": 3, "to_node_id": 6},
    {"from_node_id": 4, "to_node_id": 6},
    {"from_node_id": 5, "to_node_id": 6},
    {"from_node_id": 6, "to_node_id": 7},
    {"from_node_id": 6, "to_node_id": 8}
  ]
}"""


def load_prompt() -> str:
    return (PROMPTS_DIR / "analyze_aggregation.txt").read_text(encoding="utf-8")


def _aggregation_examples(target: Path) -> list[tuple[Path, str]]:
    """One canonical aggregation example with its expected JSON output.
    Skipped if the target is the example itself."""
    if target.resolve() == EXAMPLE_AGGREGATION_IMG.resolve():
        return []
    caption = (
        "Reference example — the image above is an aggregation floor plan "
        "with 6 apartments (A, A, B, B, A, A) around a central core "
        "containing 1 staircase and 2 elevators. The expected JSON output "
        "for this plan is:\n" + _EXAMPLE_AGGREGATION_JSON +
        "\n\nNow analyze the NEXT image below following the same schema."
    )
    return [(EXAMPLE_AGGREGATION_IMG, caption)]


def analyze_aggregation(
    client: VLMClient, image_path: Path
) -> dict | None:
    """
    Analyze an aggregation floor plan to extract apartments, stairs,
    elevators, and their connections.

    Returns:
        Parsed dict with 'nodes' and 'edges', or None on failure.
    """
    prompt = load_prompt()
    examples = _aggregation_examples(image_path)
    response = client.query(image_path, prompt, example_images=examples)
    result = extract_json(response)

    if result is None:
        logger.warning(f"Failed to parse aggregation for {image_path.name}. Retrying...")
        retry_prompt = (
            f"Your previous response was not valid JSON. "
            f"Please try again.\n\n{prompt}"
        )
        response = client.query(image_path, retry_prompt, example_images=examples)
        result = extract_json(response)

    if result is None:
        logger.error(f"Failed to extract aggregation for {image_path.name} after retry.")
        return None

    # Normalize
    if "nodes" not in result or "edges" not in result:
        logger.error(f"Missing 'nodes' or 'edges' in response for {image_path.name}")
        return None

    result["nodes"] = _normalize_nodes(result["nodes"])
    result["edges"] = _normalize_edges(result["edges"], result["nodes"])

    return result


def _normalize_nodes(nodes: list[dict]) -> list[dict]:
    """Clean and normalize extracted nodes."""
    normalized = []
    for i, node in enumerate(nodes):
        node_type = node.get("type", "apartment").lower()
        if node_type not in ("apartment", "staircase", "elevator"):
            node_type = "apartment"

        # Unify position field
        position = node.get("door_position") or node.get("position")
        if position and isinstance(position, list) and len(position) == 2:
            try:
                position = [float(position[0]), float(position[1])]
            except (ValueError, TypeError):
                position = None
        else:
            position = None

        normalized.append({
            "node_id": i,
            "type": node_type,
            "label": node.get("label"),
            "position": position,
        })

    return normalized


def _normalize_edges(edges: list[dict], nodes: list[dict]) -> list[dict]:
    """Clean and validate extracted edges."""
    valid_ids = {n["node_id"] for n in nodes}
    normalized = []
    seen = set()

    for edge in edges:
        from_id = edge.get("from_node_id", edge.get("from"))
        to_id = edge.get("to_node_id", edge.get("to"))

        if from_id is None or to_id is None:
            continue

        try:
            from_id = int(from_id)
            to_id = int(to_id)
        except (ValueError, TypeError):
            continue

        if from_id == to_id:
            continue
        if from_id not in valid_ids or to_id not in valid_ids:
            continue

        edge_key = (min(from_id, to_id), max(from_id, to_id))
        if edge_key in seen:
            continue
        seen.add(edge_key)

        normalized.append({
            "from_node_id": from_id,
            "to_node_id": to_id,
        })

    return normalized


def build_aggregation_graph(
    source_file: str,
    dataset: str,
    analysis: dict,
) -> dict:
    """
    Assemble the aggregation graph and run validation.
    """
    graph = {
        "source_file": source_file,
        "dataset": dataset,
        "type": "aggregation",
        "nodes": analysis["nodes"],
        "edges": analysis["edges"],
        "metadata": {},
        "flags": [],
    }

    # Compute metadata
    type_counts = {}
    for n in graph["nodes"]:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

    graph["metadata"] = {
        "num_apartments": type_counts.get("apartment", 0),
        "num_staircases": type_counts.get("staircase", 0),
        "num_elevators": type_counts.get("elevator", 0),
        "num_edges": len(graph["edges"]),
    }

    # Validation
    graph["flags"] = _validate(graph)
    return graph


def _validate(graph: dict) -> list[dict]:
    """Run validation rules on an aggregation graph."""
    flags = []
    nodes = graph["nodes"]
    edges = graph["edges"]

    apartments = [n for n in nodes if n["type"] == "apartment"]
    stairs = [n for n in nodes if n["type"] == "staircase"]
    elevators = [n for n in nodes if n["type"] == "elevator"]

    # Rule 1: Must have at least 2 apartments (otherwise it's not aggregation)
    if len(apartments) < 2:
        flags.append({
            "severity": "warning",
            "rule": "few_apartments",
            "detail": f"Only {len(apartments)} apartment(s) found — expected >= 2 for aggregation.",
        })

    # Rule 2: Should have at least 1 staircase or elevator
    if len(stairs) == 0 and len(elevators) == 0:
        flags.append({
            "severity": "warning",
            "rule": "no_vertical_circulation",
            "detail": "No staircase or elevator found in aggregation block.",
        })

    # Rule 3: Connectivity — all nodes should be reachable
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["node_id"])
    for e in edges:
        G.add_edge(e["from_node_id"], e["to_node_id"])

    components = list(nx.connected_components(G))
    if len(components) > 1:
        flags.append({
            "severity": "warning",
            "rule": "disconnected_graph",
            "detail": f"Graph has {len(components)} disconnected components.",
        })

    # Rule 4: Isolated apartments (no connections at all)
    connected_ids = set()
    for e in edges:
        connected_ids.add(e["from_node_id"])
        connected_ids.add(e["to_node_id"])

    for apt in apartments:
        if apt["node_id"] not in connected_ids:
            flags.append({
                "severity": "warning",
                "rule": "isolated_apartment",
                "detail": f"Apartment {apt['node_id']} (label={apt.get('label')}) has no connections.",
            })

    return flags


def save_graph(graph: dict, output_dir: Path | None = None) -> Path:
    """Save aggregation graph to JSON."""
    out_dir = output_dir or GRAPHS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Path(graph["source_file"]).stem
    out_path = out_dir / f"{source}_aggregation.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    logger.info(f"Aggregation graph saved to {out_path}")
    return out_path
