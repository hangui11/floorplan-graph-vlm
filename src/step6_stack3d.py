"""
Step 6: Concatenate 2D aggregation graphs into a 3D building graph.

For now, a single 2D graph is stacked N times (default 3) to simulate an
N-floor building. Floors are connected via their staircase and elevator
nodes: the staircase on floor k is linked to the staircase on floor k+1,
same for elevators. Apartments keep their per-floor identity (no cross-
floor apartment edges).

Later, when per-floor graphs from different images become available,
stack_graphs() can combine them directly instead of replicating one.
"""
import json
import logging
from pathlib import Path

import networkx as nx

from src.config import GRAPHS_DIR

logger = logging.getLogger(__name__)


# ── Building a 3D graph from one or more 2D graphs ─────────────────────────────

def stack_graph(
    graph_2d: dict,
    n_floors: int = 3,
) -> dict:
    """
    Replicate a single 2D graph `n_floors` times and connect them via
    staircase and elevator nodes.

    Returns:
        A 3D building graph dict.
    """
    return stack_graphs([graph_2d] * n_floors)


def stack_graphs(floor_graphs: list[dict]) -> dict:
    """
    Concatenate a list of 2D graphs (one per floor) into a 3D graph.

    Horizontal edges: the original edges from each 2D graph.
    Vertical edges: staircase-to-staircase and elevator-to-elevator
                    between adjacent floors.

    Node IDs are globalized:  new_id = floor * 1_000_000 + original_id
    so you can recover (floor, original_id) from any global id.
    """
    nodes_3d: list[dict] = []
    edges_3d: list[dict] = []

    # Per-floor remap of old -> new (global) node IDs
    remaps: list[dict[int, int]] = []

    for floor_idx, g in enumerate(floor_graphs):
        remap: dict[int, int] = {}
        offset = floor_idx * 1_000_000

        for node in g["nodes"]:
            new_id = offset + node["node_id"]
            remap[node["node_id"]] = new_id

            new_node = {
                "node_id": new_id,
                "floor": floor_idx,
                "type": node["type"],
                "label": node.get("label"),
                "position": node.get("position"),
            }
            if "details" in node:
                new_node["details"] = node["details"]
            nodes_3d.append(new_node)

        # Copy horizontal edges, remapped
        for e in g["edges"]:
            edges_3d.append({
                "from_node_id": remap[e["from_node_id"]],
                "to_node_id": remap[e["to_node_id"]],
                "edge_type": "horizontal",
                "floor": floor_idx,
            })

        remaps.append(remap)

    # Vertical edges: connect staircase→staircase and elevator→elevator
    # between consecutive floors.
    n_vertical = 0
    for floor_idx in range(len(floor_graphs) - 1):
        lower = floor_graphs[floor_idx]
        upper = floor_graphs[floor_idx + 1]

        lower_stairs    = [n for n in lower["nodes"] if n["type"] == "staircase"]
        upper_stairs    = [n for n in upper["nodes"] if n["type"] == "staircase"]
        lower_elevators = [n for n in lower["nodes"] if n["type"] == "elevator"]
        upper_elevators = [n for n in upper["nodes"] if n["type"] == "elevator"]

        # Match connectors 1-to-1 by position, falling back to order
        stair_pairs    = _match_by_position(lower_stairs,    upper_stairs)
        elevator_pairs = _match_by_position(lower_elevators, upper_elevators)

        for low_node, up_node in stair_pairs + elevator_pairs:
            edges_3d.append({
                "from_node_id": remaps[floor_idx][low_node["node_id"]],
                "to_node_id":   remaps[floor_idx + 1][up_node["node_id"]],
                "edge_type": "vertical",
                "via": low_node["type"],
            })
            n_vertical += 1

    source = floor_graphs[0].get("source_file", "unknown")
    dataset = floor_graphs[0].get("dataset", "unknown")

    # Aggregate metadata
    type_counts = {}
    for n in nodes_3d:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

    building_3d = {
        "source_file": source,
        "dataset": dataset,
        "type": "building_3d",
        "n_floors": len(floor_graphs),
        "nodes": nodes_3d,
        "edges": edges_3d,
        "metadata": {
            "num_apartments":  type_counts.get("apartment", 0),
            "num_staircases":  type_counts.get("staircase", 0),
            "num_elevators":   type_counts.get("elevator", 0),
            "num_horizontal_edges": sum(1 for e in edges_3d if e["edge_type"] == "horizontal"),
            "num_vertical_edges":   n_vertical,
        },
    }

    building_3d["flags"] = _validate_3d(building_3d)
    return building_3d


def _match_by_position(
    lower: list[dict],
    upper: list[dict],
) -> list[tuple[dict, dict]]:
    """
    Pair lower-floor connectors with upper-floor connectors.

    Prefers nearest-position matches. If either list is empty, returns [].
    Unmatched extras are dropped.
    """
    if not lower or not upper:
        return []

    pairs = []
    used_upper = set()

    for low in lower:
        best_idx = None
        best_dist = float("inf")
        for i, up in enumerate(upper):
            if i in used_upper:
                continue
            dist = _position_distance(low.get("position"), up.get("position"))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            used_upper.add(best_idx)
            pairs.append((low, upper[best_idx]))

    return pairs


def _position_distance(p1, p2) -> float:
    """Euclidean distance between two [x, y] positions; inf if either is None."""
    if not p1 or not p2:
        return 0.0  # When positions are unknown, treat as co-located
    try:
        dx = float(p1[0]) - float(p2[0])
        dy = float(p1[1]) - float(p2[1])
        return (dx * dx + dy * dy) ** 0.5
    except (TypeError, ValueError, IndexError):
        return 0.0


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_3d(building: dict) -> list[dict]:
    flags = []
    vertical = [e for e in building["edges"] if e["edge_type"] == "vertical"]

    if building["n_floors"] > 1 and not vertical:
        flags.append({
            "severity": "error",
            "rule": "no_vertical_edges",
            "detail": "Multi-floor building has no vertical (stair/elevator) edges between floors.",
        })

    # Connectivity of the 3D graph
    G = nx.Graph()
    for n in building["nodes"]:
        G.add_node(n["node_id"])
    for e in building["edges"]:
        G.add_edge(e["from_node_id"], e["to_node_id"])

    n_components = nx.number_connected_components(G)
    if n_components > 1:
        flags.append({
            "severity": "warning",
            "rule": "disconnected_building",
            "detail": f"3D graph has {n_components} disconnected components.",
        })

    return flags


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_3d_graph(building: dict, output_dir: Path | None = None) -> Path:
    """Save a 3D building graph to JSON."""
    out_dir = output_dir or GRAPHS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Path(building["source_file"]).stem
    out_path = out_dir / f"{source}_3d.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(building, f, indent=2, ensure_ascii=False)

    logger.info(f"3D graph saved to {out_path}")
    return out_path


def stack_all_graphs(
    n_floors: int = 3,
    graphs_dir: Path | None = None,
) -> list[Path]:
    """
    For every 2D aggregation graph in the graphs directory, produce a
    stacked 3D version with n_floors copies, connected via stairs/elevators.

    Returns the list of output paths.
    """
    gdir = graphs_dir or GRAPHS_DIR
    graph_files = sorted(gdir.glob("*_aggregation.json"))
    logger.info(f"Stacking {len(graph_files)} graphs into 3D ({n_floors} floors each)...")

    outputs = []
    for gf in graph_files:
        with open(gf, "r", encoding="utf-8") as f:
            graph_2d = json.load(f)

        building = stack_graph(graph_2d, n_floors=n_floors)
        out = save_3d_graph(building)
        outputs.append(out)

        meta = building["metadata"]
        logger.info(
            f"  {gf.stem}: {meta['num_horizontal_edges']} horizontal + "
            f"{meta['num_vertical_edges']} vertical edges"
        )

    return outputs
