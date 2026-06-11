"""
Step 4: Automated validation of aggregation graphs.

Runs structural, topological, and consistency checks. Produces a per-graph
validation report and marks graphs that need human review (Step 5).
"""
import json
import logging
from collections import Counter
from pathlib import Path

import networkx as nx

from src.config import GRAPHS_DIR, STATS_DIR, LOGS_DIR

logger = logging.getLogger(__name__)


# ── Validation scoring ─────────────────────────────────────────────────────────

def validate_graph(graph: dict) -> dict:
    """
    Run all validation checks on a graph. Returns a structured report.

    Report schema:
        {
          "source_file": ...,
          "passed": bool,              # True if all hard checks pass
          "needs_human_review": bool,  # True if missing stairs/elevators or isolated
          "score": float,              # 0..1 quality score
          "checks": {
            "has_apartments":   {"passed": bool, "value": int},
            "has_connectors":   {"passed": bool, "has_stairs": bool, "has_elevator": bool},
            "all_connected":    {"passed": bool, "n_components": int},
            "no_isolated_apt":  {"passed": bool, "isolated": [...]},
            "every_apt_reaches_connector": {"passed": bool, "unreachable": [...]},
            "no_dup_positions": {"passed": bool, "duplicates": [...]}
          }
        }
    """
    nodes = graph["nodes"]
    edges = graph["edges"]

    apartments = [n for n in nodes if n["type"] == "apartment"]
    stairs = [n for n in nodes if n["type"] == "staircase"]
    elevators = [n for n in nodes if n["type"] == "elevator"]
    connector_ids = {n["node_id"] for n in stairs + elevators}

    checks = {}

    # Check 1: has at least 2 apartments
    checks["has_apartments"] = {
        "passed": len(apartments) >= 2,
        "value": len(apartments),
    }

    # Check 2: has at least one vertical connector
    checks["has_connectors"] = {
        "passed": (len(stairs) + len(elevators)) >= 1,
        "has_stairs": len(stairs) > 0,
        "has_elevator": len(elevators) > 0,
    }

    # Build nx graph for structural checks
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["node_id"], type=n["type"])
    for e in edges:
        G.add_edge(e["from_node_id"], e["to_node_id"])

    # Check 3: all nodes connected (single component)
    if len(nodes) == 0:
        checks["all_connected"] = {"passed": False, "n_components": 0}
    else:
        n_components = nx.number_connected_components(G)
        checks["all_connected"] = {
            "passed": n_components == 1,
            "n_components": n_components,
        }

    # Check 4: no isolated apartments
    isolated = [
        apt["node_id"] for apt in apartments
        if G.degree(apt["node_id"]) == 0
    ]
    checks["no_isolated_apt"] = {
        "passed": len(isolated) == 0,
        "isolated": isolated,
    }

    # Check 5: every apartment can reach a stair or elevator
    unreachable = []
    if connector_ids:
        for apt in apartments:
            apt_id = apt["node_id"]
            if not any(
                nx.has_path(G, apt_id, cid) for cid in connector_ids if cid in G.nodes
            ):
                unreachable.append(apt_id)
        checks["every_apt_reaches_connector"] = {
            "passed": len(unreachable) == 0,
            "unreachable": unreachable,
        }
    else:
        checks["every_apt_reaches_connector"] = {
            "passed": False,
            "unreachable": [apt["node_id"] for apt in apartments],
        }

    # Check 6: no duplicate positions (apartments at the same [x, y])
    pos_to_nodes = {}
    duplicates = []
    for n in nodes:
        pos = n.get("position")
        if pos and isinstance(pos, list) and len(pos) == 2:
            key = (round(pos[0], 1), round(pos[1], 1))
            pos_to_nodes.setdefault(key, []).append(n["node_id"])
    for key, ids in pos_to_nodes.items():
        if len(ids) > 1:
            duplicates.append({"position": list(key), "node_ids": ids})
    checks["no_dup_positions"] = {
        "passed": len(duplicates) == 0,
        "duplicates": duplicates,
    }

    # Aggregate
    hard_checks = [
        "has_apartments",
        "all_connected",
        "no_isolated_apt",
    ]
    soft_checks = [
        "has_connectors",
        "every_apt_reaches_connector",
        "no_dup_positions",
    ]

    passed = all(checks[k]["passed"] for k in hard_checks)
    total_checks = len(hard_checks) + len(soft_checks)
    score = sum(
        1 for k in hard_checks + soft_checks if checks[k]["passed"]
    ) / total_checks

    # Needs human review if connection components missing or unreachable
    needs_human_review = (
        not checks["has_connectors"]["passed"]
        or not checks["every_apt_reaches_connector"]["passed"]
    )

    return {
        "source_file": graph["source_file"],
        "dataset": graph.get("dataset"),
        "passed": passed,
        "needs_human_review": needs_human_review,
        "score": round(score, 3),
        "checks": checks,
    }


def validate_all(graphs_dir: Path | None = None) -> dict:
    """
    Validate every aggregation graph in the graphs directory and produce
    an aggregate report.
    """
    gdir = graphs_dir or GRAPHS_DIR
    graph_files = sorted(gdir.glob("*_aggregation.json"))
    logger.info(f"Validating {len(graph_files)} graphs...")

    reports = []
    needs_review = []
    failing = []

    for gf in graph_files:
        with open(gf, "r", encoding="utf-8") as f:
            graph = json.load(f)

        report = validate_graph(graph)
        reports.append(report)

        if report["needs_human_review"]:
            needs_review.append(report["source_file"])
        if not report["passed"]:
            failing.append(report["source_file"])

    # Aggregate statistics
    rule_counts = Counter()
    for r in reports:
        for rule, result in r["checks"].items():
            if not result["passed"]:
                rule_counts[rule] += 1

    summary = {
        "total_graphs": len(reports),
        "passing": sum(1 for r in reports if r["passed"]),
        "failing": len(failing),
        "needing_human_review": len(needs_review),
        "avg_score": round(sum(r["score"] for r in reports) / max(len(reports), 1), 3),
        "failed_rules": dict(rule_counts.most_common()),
        "graphs_needing_review": needs_review,
        "graphs_failing": failing,
    }

    # Save outputs
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(STATS_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(LOGS_DIR / "validation_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    # Save list of graphs needing human review (for Step 5)
    with open(LOGS_DIR / "needs_human_review.json", "w", encoding="utf-8") as f:
        json.dump(needs_review, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Validation complete: {summary['passing']}/{summary['total_graphs']} passed. "
        f"{summary['needing_human_review']} need human review."
    )
    return summary
