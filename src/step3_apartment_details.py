"""
Step 3: Enrich each apartment node with interior details.

For every 'apartment' node in the aggregation graph, ask the VLM to
count the number of bedrooms, bathrooms, living rooms, etc. inside it.
"""
import json
import logging
from pathlib import Path

from src.config import PROMPTS_DIR, GRAPHS_DIR
from src.vlm_client import VLMClient
from src.utils.json_parser import extract_json

logger = logging.getLogger(__name__)


def load_prompt() -> str:
    return (PROMPTS_DIR / "apartment_details.txt").read_text(encoding="utf-8")


def extract_apartment_details(
    client: VLMClient,
    image_path: Path,
    apartment_node: dict,
    prompt_template: str | None = None,
) -> dict | None:
    """
    Ask the VLM to describe the interior of a single apartment.

    Args:
        client: Loaded VLMClient.
        image_path: Path to the aggregation floor plan image.
        apartment_node: A node dict with type='apartment'.
        prompt_template: Optional cached prompt template.

    Returns:
        Dict of interior details, or None on failure.
    """
    if prompt_template is None:
        prompt_template = load_prompt()

    label = apartment_node.get("label") or f"Apartment {apartment_node['node_id']}"
    position = apartment_node.get("position") or [50, 50]
    x, y = position[0], position[1]

    prompt = (
        prompt_template
        .replace("{apartment_label}", str(label))
        .replace("{x}", str(x))
        .replace("{y}", str(y))
    )

    response = client.query(image_path, prompt)
    result = extract_json(response)

    if result is None:
        logger.warning(
            f"Failed to parse apartment details for {image_path.name} "
            f"apartment {apartment_node['node_id']} ({label})"
        )
        return None

    return _normalize_details(result)


def _normalize_details(details: dict) -> dict:
    """Normalize and type-cast the extracted apartment details."""
    normalized = {
        "num_bedrooms": _safe_int(details.get("num_bedrooms")),
        "num_bathrooms": _safe_int(details.get("num_bathrooms")),
        "num_kitchens": _safe_int(details.get("num_kitchens")),
        "num_living_rooms": _safe_int(details.get("num_living_rooms")),
        "num_terraces": _safe_int(details.get("num_terraces")),
        "has_corridor": bool(details.get("has_corridor")) if details.get("has_corridor") is not None else None,
        "total_rooms": _safe_int(details.get("total_rooms")),
        "room_labels": details.get("room_labels") or [],
    }

    # Sanity: total_rooms should equal sum of counted rooms
    component_sum = sum(
        v for v in [
            normalized["num_bedrooms"],
            normalized["num_bathrooms"],
            normalized["num_living_rooms"],
            normalized["num_terraces"],
        ] if v is not None
    )
    if normalized["total_rooms"] is None and component_sum > 0:
        normalized["total_rooms"] = component_sum

    return normalized


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def enrich_graph(
    client: VLMClient,
    image_path: Path,
    graph: dict,
) -> dict:
    """
    Add interior details to every apartment node in the graph.

    Modifies graph in-place and returns it.
    """
    prompt_template = load_prompt()
    enriched_count = 0

    for node in graph["nodes"]:
        if node["type"] != "apartment":
            continue

        details = extract_apartment_details(
            client, image_path, node, prompt_template=prompt_template
        )
        if details is None:
            node["details"] = None
            continue

        node["details"] = details
        enriched_count += 1
        logger.info(
            f"  Apartment {node['node_id']} ({node.get('label')}): "
            f"bedrooms={details['num_bedrooms']}, "
            f"bathrooms={details['num_bathrooms']}, "
            f"total={details['total_rooms']}"
        )

    graph["metadata"]["apartments_with_details"] = enriched_count
    return graph


def save_enriched_graph(graph: dict, output_dir: Path | None = None) -> Path:
    """Save an enriched graph to JSON."""
    out_dir = output_dir or GRAPHS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Path(graph["source_file"]).stem
    out_path = out_dir / f"{source}_aggregation.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    logger.info(f"Enriched graph saved to {out_path}")
    return out_path
