"""
Classification evaluation: compare Step 1 predictions against hand-labelled
ground truth in outputs/stats/classification_eval.csv.

The CSV must have columns: file, dataset, predicted, truth
(truth filled in manually; rows with an empty truth are skipped).

Prints overall accuracy, scikit-learn's classification_report (per-class
precision / recall / F1 / support), and a predicted-vs-truth confusion matrix.

Usage:
    python -m src.eval_classification
    python -m src.eval_classification --csv path/to/file.csv
"""
import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import STATS_DIR

LABELS = ["aggregation", "individual", "other"]


def load_labelled(csv_path: Path):
    """Return (y_true, y_pred) from rows that have both predicted and truth filled."""
    y_true, y_pred = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        t = r.get("truth", "").strip()
        p = r.get("predicted", "").strip()
        if t and p:
            y_true.append(t)
            y_pred.append(p)
    return y_true, y_pred, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Step 1 classification accuracy")
    parser.add_argument("--csv", type=str,
                        default=str(STATS_DIR / "classification_eval.csv"),
                        help="Path to the labelled CSV (file,dataset,predicted,truth)")
    parser.add_argument("--save", type=str, nargs="?",
                        const=str(STATS_DIR / "classification_report.txt"),
                        default=None,
                        help="Also write the report to this file "
                             "(default: outputs/stats/classification_report.txt)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    y_true, y_pred, total_rows = load_labelled(csv_path)
    if not y_true:
        raise SystemExit("No labelled rows (truth column empty).")

    # Use the fixed label order, plus any unexpected labels that appear.
    labels = list(LABELS)
    for v in set(y_true) | set(y_pred):
        if v not in labels:
            labels.append(v)

    lines = []
    lines.append("=" * 60)
    lines.append("STEP 1 CLASSIFICATION EVALUATION")
    lines.append("=" * 60)
    lines.append(f"Labelled samples : {len(y_true)}  (of {total_rows} rows)")
    lines.append(f"Accuracy         : {accuracy_score(y_true, y_pred):.4f}")
    lines.append("")
    lines.append("Classification report:")
    lines.append(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    lines.append("Confusion matrix (rows = truth, cols = predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    lines.append("  " + "".join(f"{l[:12]:>14}" for l in labels))
    for i, l in enumerate(labels):
        lines.append(f"  {l:<12}" + "".join(f"{cm[i][j]:>14}" for j in range(len(labels))))

    report = "\n".join(lines)
    print("\n" + report + "\n")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
