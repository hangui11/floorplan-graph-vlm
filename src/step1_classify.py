"""
Step 1: Classify each image as 'aggregation' or 'individual'.

- aggregation: Full building floor with multiple apartments + shared circulation
- individual: Single apartment/housing unit layout
"""
import json
import logging
import re
from pathlib import Path

from src.config import (
    DATASETS,
    PROMPTS_DIR,
    LOGS_DIR,
    EXAMPLE_AGGREGATION_IMG,
    EXAMPLE_INDIVIDUAL_IMG,
)
from src.vlm_client import VLMClient

logger = logging.getLogger(__name__)


def load_prompt() -> str:
    return (PROMPTS_DIR / "classify_image.txt").read_text(encoding="utf-8")


def _classification_examples(target: Path) -> list[tuple[Path, str]]:
    """One canonical example per class, prepended before the target image.
    Skips an example if it is the same file as the target."""
    candidates = [
        (EXAMPLE_AGGREGATION_IMG,
         "Reference example — this plan is an AGGREGATION: multiple apartments "
         "(labeled A, A, B, B) arranged around a shared central core with a "
         "staircase and two elevators. Answer for a plan like this: aggregation"),
        (EXAMPLE_INDIVIDUAL_IMG,
         "Reference example — this plan is an INDIVIDUAL apartment: a single "
         "dwelling with one kitchen (cuina), one bathroom (bany), one "
         "living-dining (estar menjador), and two bedrooms (dormitori). "
         "Answer for a plan like this: individual"),
    ]
    target_resolved = target.resolve()
    return [(p, c) for p, c in candidates if p.resolve() != target_resolved]


def classify_image(client: VLMClient, image_path: Path, prompt: str) -> str:
    """
    Classify a single image as 'aggregation' or 'individual'.

    Returns:
        'aggregation', 'individual', or 'unknown'.
    """
    response = client.query(
        image_path,
        prompt,
        example_images=_classification_examples(image_path),
        deterministic=True,
    ).strip().lower()

    # Extract the classification word from the response
    if "aggregation" in response:
        return "aggregation"
    elif "individual" in response:
        return "individual"
    else:
        logger.warning(
            f"Could not parse classification for {image_path.name}: {response!r}"
        )
        return "unknown"


def classify_dataset(
    client: VLMClient,
    dataset_name: str | None = None,
    save_log: bool = True,
) -> dict[str, list[Path]]:
    """
    Classify all images in the dataset(s).

    Returns:
        Dict with keys 'aggregation', 'individual', 'unknown',
        each a list of image Paths.
    """
    prompt = load_prompt()
    results = {"aggregation": [], "individual": [], "unknown": []}
    log_entries = []

    datasets = {dataset_name: DATASETS[dataset_name]} if dataset_name else DATASETS

    for ds_name, ds_path in datasets.items():
        image_files = sorted(ds_path.glob("*.jpg"))
        logger.info(f"Classifying {ds_name}: {len(image_files)} images")

        for img_path in image_files:
            label = classify_image(client, img_path, prompt)
            results[label].append(img_path)

            log_entries.append({
                "file": str(img_path),
                "dataset": ds_name,
                "classification": label,
            })
            logger.info(f"  {img_path.name} -> {label}")

    if save_log:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / "step1_classification.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)
        logger.info(f"Classification log saved to {log_path}")

    logger.info(
        f"Classification results: "
        f"{len(results['aggregation'])} aggregation, "
        f"{len(results['individual'])} individual, "
        f"{len(results['unknown'])} unknown"
    )

    return results
