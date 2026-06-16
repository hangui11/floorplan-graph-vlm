"""
Build a 3D building graph from a chosen set of 2D aggregation graphs.

Unlike Step 6's default (which replicates ONE floor N times), this takes
several DIFFERENT *_aggregation.json files and stacks them as consecutive
floors — floor 0 = first file given, floor 1 = second, and so on. Useful for
assembling a genuine multi-floor building from per-floor plans.

The connectors (staircase/elevator) of adjacent floors are linked by nearest
position, exactly as in Step 6.

Usage:
    python -m src.stack_custom inca_555 inca_580
    python -m src.stack_custom inca_555 inca_580 --out my_building --no-vis
"""
import argparse
import json
from pathlib import Path

from src.config import GRAPHS_DIR, VIS_DIR
from src.step6_stack3d import stack_graphs, save_3d_graph


def _resolve_graph(name: str) -> Path:
    """Accept 'inca_555', 'inca_555_aggregation', or a full path."""
    p = Path(name)
    if p.exists():
        return p
    stem = p.stem
    if not stem.endswith("_aggregation"):
        stem = f"{stem}_aggregation"
    candidate = GRAPHS_DIR / f"{stem}.json"
    if candidate.exists():
        return candidate
    raise SystemExit(f"Graph not found for '{name}' (looked for {candidate})")


def main():
    parser = argparse.ArgumentParser(
        description="Stack chosen 2D aggregation graphs into one 3D building graph"
    )
    parser.add_argument("graphs", nargs="+",
                        help="2+ aggregation graphs (names or paths), one per floor, bottom-up")
    parser.add_argument("--out", type=str, default=None,
                        help="Output stem (default: derived from the first graph)")
    parser.add_argument("--no-vis", action="store_true", help="Skip the 3D PNG render")
    args = parser.parse_args()

    if len(args.graphs) < 2:
        raise SystemExit("Provide at least 2 graphs to stack (one per floor).")

    paths = [_resolve_graph(g) for g in args.graphs]
    floor_graphs = []
    for i, p in enumerate(paths):
        with open(p, "r", encoding="utf-8") as f:
            g = json.load(f)
        floor_graphs.append(g)
        m = g.get("metadata", {})
        print(f"  floor {i}: {p.name}  "
              f"(apts={m.get('num_apartments',0)}, "
              f"stairs={m.get('num_staircases',0)}, elev={m.get('num_elevators',0)})")

    building = stack_graphs(floor_graphs)

    # Name the output after --out, or the first source file.
    out_stem = args.out or (Path(floor_graphs[0]["source_file"]).stem + "_custom")
    building["source_file"] = f"{out_stem}.jpg"

    out_path = save_3d_graph(building)
    meta = building["metadata"]
    print(f"\n3D graph: {building['n_floors']} floors, "
          f"{meta['num_apartments']} apartments, "
          f"{meta['num_horizontal_edges']} horizontal + "
          f"{meta['num_vertical_edges']} vertical edges")
    print(f"Saved: {out_path}")

    if not args.no_vis:
        try:
            from src.utils.visualization import draw_3d_graph
            VIS_DIR.mkdir(parents=True, exist_ok=True)
            vis_path = VIS_DIR / f"{out_stem}_3d.png"
            draw_3d_graph(
                building, save_path=vis_path,
                title=f"Custom 3D building ({building['n_floors']} floors)",
            )
            print(f"Viz:   {vis_path}")
        except Exception as e:
            print(f"  (visualization skipped: {e})")


if __name__ == "__main__":
    main()
