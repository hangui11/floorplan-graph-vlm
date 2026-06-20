"""
Manual spot-check evaluation: summarise the hand-filled extraction spot-check
in outputs/stats/spotcheck_sample.csv.

The CSV must have columns:
    dataset, file, true_class, missing_nodes, spurious_nodes,
    wrong_edges, wrong_rooms, fully_correct, notes

Misclassified plans (true_class != "aggregation") are reported separately as
classification leakage and excluded from the extraction-quality figures, which
are computed only over the true-aggregation graphs.

Usage:
    python -m src.eval_spotcheck
    python -m src.eval_spotcheck --csv path/to/file.csv --save
"""
import argparse
import csv
from pathlib import Path

from src.config import STATS_DIR

ERROR_COLS = [
    ("missing_nodes", "Missing nodes (apartments / connectors)"),
    ("spurious_nodes", "Spurious nodes"),
    ("wrong_edges", "Incorrect edges"),
    ("wrong_rooms", "Incorrect room inventory"),
]


def _num(v: str) -> int:
    """Parse a cell as a non-negative integer; blanks and junk count as 0."""
    v = (v or "").strip()
    return int(v) if v.lstrip("-").isdigit() else 0


def evaluate(rows: list[dict]) -> dict:
    agg = [r for r in rows if r.get("true_class", "").strip() == "aggregation"]
    mis = [r for r in rows if r.get("true_class", "").strip() not in ("", "aggregation")]

    fully_correct = [
        r for r in agg if r.get("fully_correct", "").strip().lower() == "yes"
    ]

    per_cat = {}
    for col, label in ERROR_COLS:
        affected = [r for r in agg if _num(r.get(col, "")) > 0]
        total = sum(_num(r.get(col, "")) for r in agg)
        per_cat[col] = {
            "label": label,
            "affected": len(affected),
            "instances": total,
        }

    any_error = [
        r for r in agg
        if any(_num(r.get(c, "")) > 0 for c, _ in ERROR_COLS)
    ]

    return {
        "total": len(rows),
        "misclassified": mis,
        "agg": agg,
        "fully_correct": len(fully_correct),
        "any_error": len(any_error),
        "per_cat": per_cat,
    }


def format_report(res: dict) -> str:
    n_agg = len(res["agg"])
    pct = (lambda x: f"{(100*x/n_agg):.1f}%" if n_agg else "n/a")

    lines = []
    lines.append("=" * 64)
    lines.append("EXTRACTION SPOT-CHECK")
    lines.append("=" * 64)
    lines.append(f"Sampled plans              : {res['total']}")
    lines.append(f"Misclassified (leakage)    : {len(res['misclassified'])}")
    lines.append(f"True-aggregation graphs    : {n_agg}")
    lines.append("")
    lines.append(f"Fully correct              : {res['fully_correct']} ({pct(res['fully_correct'])})")
    lines.append(f"With >= 1 semantic error   : {res['any_error']} ({pct(res['any_error'])})")
    lines.append("")
    lines.append("Error breakdown (over the true-aggregation graphs):")
    lines.append(f"  {'category':<42}{'graphs':>8}{'instances':>11}")
    for col, _ in ERROR_COLS:
        c = res["per_cat"][col]
        graphs = f"{c['affected']} ({pct(c['affected'])})"
        lines.append(f"  {c['label']:<42}{graphs:>8}{c['instances']:>11}")
    lines.append("")
    lines.append("Note: a graph may exhibit several error categories, so the "
                 "rows are not mutually exclusive.")

    if res["misclassified"]:
        lines.append("")
        lines.append("Classification leakage (true_class != aggregation):")
        for r in res["misclassified"]:
            lines.append(f"  {r['file']}  (true_class={r['true_class']})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarise the extraction spot-check")
    parser.add_argument("--csv", type=str,
                        default=str(STATS_DIR / "spotcheck_sample.csv"),
                        help="Path to the filled spot-check CSV")
    parser.add_argument("--save", type=str, nargs="?",
                        const=str(STATS_DIR / "spotcheck_report.txt"),
                        default=None,
                        help="Also write the report to this file "
                             "(default: outputs/stats/spotcheck_report.txt)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV is empty.")

    res = evaluate(rows)
    report = format_report(res)
    print("\n" + report + "\n")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
