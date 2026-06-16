"""
End-to-end pipeline orchestrator.

Steps:
  1. Classify images as 'aggregation' vs 'individual'
  2. For aggregation images, extract a 2D graph (apartments, stairs, elevators)
  3. Enrich each apartment node with interior details (room counts)
  4. Automated validation (structural/topological checks)
  5. (Interactive) Human-in-the-loop refinement for graphs missing stairs/elevators
  6. Stack each 2D graph into a 3D graph (default 3 floors)

Steps 1-4 and 6 are automatic. Step 5 is invoked explicitly with --review.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import DATASETS, GRAPHS_DIR, VIS_DIR, LOGS_DIR
from src.vlm_client import VLMClient
from src.step1_classify import classify_image, load_prompt as load_classify_prompt
from src.step2_aggregation import (
    analyze_aggregation,
    build_aggregation_graph,
    save_graph,
)
from src.step3_apartment_details import enrich_graph, save_enriched_graph
from src.step4_validate import validate_all
from src.step5_human_review import run_interactive_human_review
from src.step6_stack3d import stack_all_graphs
from src.utils.visualization import draw_side_by_side, draw_3d_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def process_aggregation_image(
    client: VLMClient,
    image_path: Path,
    dataset: str,
    enrich: bool = True,
    visualize: bool = True,
) -> dict | None:
    """Run Steps 2-3 on a single aggregation image."""
    logger.info(f"  Analyzing: {image_path.name}")

    # Step 2
    analysis = analyze_aggregation(client, image_path)
    if analysis is None:
        logger.error(f"    Step 2 FAILED for {image_path.name}")
        return None

    graph = build_aggregation_graph(
        source_file=image_path.name,
        dataset=dataset,
        analysis=analysis,
    )
    meta = graph["metadata"]
    logger.info(
        f"    Step 2: {meta['num_apartments']} apartments, "
        f"{meta['num_staircases']} stairs, {meta['num_elevators']} elevators, "
        f"{meta['num_edges']} edges"
    )
    if graph.get("building_analysis"):
        logger.info(f"    Step 2 reasoning: {graph['building_analysis']}")

    # Step 3: apartment details
    if enrich:
        logger.info(f"    Step 3: enriching apartment details...")
        graph = enrich_graph(client, image_path, graph)

    save_enriched_graph(graph) if enrich else save_graph(graph)

    if visualize:
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        vis_path = VIS_DIR / f"{image_path.stem}_aggregation.png"
        draw_side_by_side(
            image_path, graph,
            save_path=vis_path,
            title=f"{dataset} / {image_path.name}",
        )

    return graph


def run_revalidation(
    visualize: bool = True,
    validate: bool = True,
    stack: bool = True,
    n_floors: int = 3,
):
    """
    Re-run only the non-VLM downstream steps (Step 4 validation + Step 6 3D
    stacking) against the graphs currently on disk in outputs/graphs/.

    Used standalone after Step 5 human review (`--revalidate`) to refresh the
    validation summary and 3D graphs from the corrected 2D graphs, and reused
    by run_pipeline() at the end of a full run. Touches no VLM.
    """
    # Step 4: automated validation
    if validate:
        logger.info(f"--- Step 4: automated validation ---")
        summary = validate_all()
        logger.info(
            f"  {summary['passing']}/{summary['total_graphs']} passed. "
            f"{summary['needing_human_review']} need human review."
        )
        if summary["graphs_needing_review"]:
            logger.info(f"  Plans needing review (run `--review` to fix):")
            for src in summary["graphs_needing_review"]:
                logger.info(f"    - {src}")

    # Step 6: 3D stacking
    if stack:
        logger.info(f"--- Step 6: stacking 2D graphs into 3D ({n_floors} floors) ---")
        outputs = stack_all_graphs(n_floors=n_floors)
        logger.info(f"  Wrote {len(outputs)} 3D graphs.")

        # Render 3D visualization for each
        if visualize:
            for out in outputs:
                with open(out, "r", encoding="utf-8") as f:
                    building = json.load(f)
                stem = Path(building["source_file"]).stem
                vis_path = VIS_DIR / f"{stem}_3d.png"
                try:
                    draw_3d_graph(
                        building,
                        save_path=vis_path,
                        title=f"{building['dataset']} / {stem} (3D, {n_floors} floors)",
                    )
                except Exception as e:
                    logger.warning(f"  3D viz failed for {stem}: {e}")


def run_pipeline(
    dataset_name: str | None = None,
    visualize: bool = True,
    enrich: bool = True,
    validate: bool = True,
    stack: bool = True,
    n_floors: int = 3,
    limit: int | None = None,
    model_path: str | None = None,
):
    """Run the automated pipeline (Steps 1-4 + 6)."""
    client = VLMClient(model_path=model_path)
    client.load()

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    datasets = {dataset_name: DATASETS[dataset_name]} if dataset_name else DATASETS
    classify_prompt = load_classify_prompt()

    all_classifications = []
    all_graphs = []
    failed = []

    for ds_name, ds_path in datasets.items():
        image_files = sorted(ds_path.glob("*.jpg"))
        logger.info(f"=== Dataset: {ds_name} ({len(image_files)} images) ===")

        if limit:
            image_files = image_files[:limit]

        # Step 1: classify
        logger.info(f"--- Step 1: classifying images ---")
        aggregation_images = []
        for img_path in tqdm(image_files, desc=f"Classifying {ds_name}"):
            label = classify_image(client, img_path, classify_prompt)
            all_classifications.append({
                "file": str(img_path),
                "dataset": ds_name,
                "classification": label,
            })
            logger.info(f"  {img_path.name} -> {label}")
            if label == "aggregation":
                aggregation_images.append(img_path)

        logger.info(
            f"  {len(aggregation_images)} aggregation / "
            f"{len(image_files) - len(aggregation_images)} non-aggregation "
            f"(individual / other / unknown — skipped from extraction)"
        )

        # Steps 2-3: extract and enrich
        logger.info(f"--- Steps 2-3: extracting + enriching ---")
        for img_path in tqdm(aggregation_images, desc=f"Analyzing {ds_name}"):
            graph = process_aggregation_image(
                client, img_path, ds_name,
                enrich=enrich, visualize=visualize,
            )
            if graph:
                all_graphs.append(graph)
            else:
                failed.append(str(img_path))

    # Save classification log
    with open(LOGS_DIR / "step1_classification.json", "w", encoding="utf-8") as f:
        json.dump(all_classifications, f, indent=2, ensure_ascii=False)

    # Steps 4 + 6: validation and 3D stacking (no VLM)
    run_revalidation(
        visualize=visualize,
        validate=validate,
        stack=stack,
        n_floors=n_floors,
    )

    # Summary
    n_agg = sum(1 for c in all_classifications if c["classification"] == "aggregation")
    n_ind = sum(1 for c in all_classifications if c["classification"] == "individual")
    n_oth = sum(1 for c in all_classifications if c["classification"] == "other")
    n_unk = sum(1 for c in all_classifications if c["classification"] == "unknown")

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete.")
    logger.info(f"  Total images:   {len(all_classifications)}")
    logger.info(f"  Aggregation:    {n_agg}")
    logger.info(f"  Individual:     {n_ind}")
    logger.info(f"  Other:          {n_oth}")
    logger.info(f"  Unknown:        {n_unk}")
    logger.info(f"  Graphs:         {len(all_graphs)}")
    logger.info(f"  Failed:         {len(failed)}")

    if failed:
        with open(LOGS_DIR / "failed_images.json", "w") as f:
            json.dump(failed, f, indent=2)

    return all_graphs


def main():
    parser = argparse.ArgumentParser(description="Floor Plan Aggregation Pipeline")
    parser.add_argument("--dataset", choices=["IBAVI", "IMPSOL", "INCASOL"],
                        help="Process only this dataset")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip Step 3 apartment detail enrichment")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip Step 4 automated validation")
    parser.add_argument("--no-stack", action="store_true",
                        help="Skip Step 6 3D stacking")
    parser.add_argument("--n-floors", type=int, default=3,
                        help="Number of floors for 3D stacking (default: 3)")
    parser.add_argument("--review", "--rlhf", dest="review", action="store_true",
                        help="Run the interactive human-in-the-loop refinement step only (Step 5)")
    parser.add_argument("--revalidate", action="store_true",
                        help="Re-run only Steps 4+6 (validation + 3D stacking) on the "
                             "graphs already in outputs/graphs/ — no VLM. Use after --review.")
    parser.add_argument("--limit", type=int,
                        help="Process at most N images (for testing)")
    parser.add_argument("--model-path", type=str,
                        help="Override model path")

    args = parser.parse_args()

    if args.review:
        # Interactive session only; requires prior validation output
        client = VLMClient(model_path=args.model_path)
        client.load()
        run_interactive_human_review(client, visualize=not args.no_vis)
        return

    if args.revalidate:
        # Steps 4+6 only, against the graphs already on disk. No VLM/model load.
        run_revalidation(
            visualize=not args.no_vis,
            validate=not args.no_validate,
            stack=not args.no_stack,
            n_floors=args.n_floors,
        )
        return

    run_pipeline(
        dataset_name=args.dataset,
        visualize=not args.no_vis,
        enrich=not args.no_enrich,
        validate=not args.no_validate,
        stack=not args.no_stack,
        n_floors=args.n_floors,
        limit=args.limit,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
