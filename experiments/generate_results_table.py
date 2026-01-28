"""
Generate a publication-style table image from multi-seed results.

Example:
  python experiments/generate_results_table.py \
    --input experiments/results/multiseed_results_20260105_202940.json \
    --output experiments/results/visualizations/multiseed_results_20260105_202940/table.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_results(path: Path) -> Dict:
    with path.open('r') as f:
        return json.load(f)


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}"


def _extract_row(metrics: Dict) -> Tuple[int, int, int, int, str, str, str]:
    tp = int(round(metrics.get('true_positives_mean', 0)))
    fp = int(round(metrics.get('false_positives_mean', 0)))
    tn = int(round(metrics.get('true_negatives_mean', 0)))
    fn = int(round(metrics.get('false_negatives_mean', 0)))
    precision = metrics.get('precision_mean', 0.0)
    recall = metrics.get('recall_mean', 0.0)
    f1 = metrics.get('f1_mean', 0.0)
    return tp, fp, tn, fn, _format_percent(precision), _format_percent(recall), _format_percent(f1)


def _build_rows(aggregated_results: Dict, methods: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for method in methods:
        metrics = aggregated_results.get(method)
        if not metrics:
            continue
        tp, fp, tn, fn, prec, rec, f1 = _extract_row(metrics)
        rows.append([
            method.replace('_', ' ').title(),
            str(tp), str(fp), str(tn), str(fn),
            prec, rec, f1,
        ])
    return rows


def _compute_overall(aggregated_results: Dict, methods: List[str]) -> List[str]:
    metric_keys = [
        'true_positives_mean',
        'false_positives_mean',
        'true_negatives_mean',
        'false_negatives_mean',
        'precision_mean',
        'recall_mean',
        'f1_mean',
    ]
    values = {k: [] for k in metric_keys}
    for method in methods:
        metrics = aggregated_results.get(method)
        if not metrics:
            continue
        for key in metric_keys:
            if key in metrics:
                values[key].append(metrics[key])

    def avg(key: str) -> float:
        return float(np.mean(values[key])) if values[key] else 0.0

    tp = int(round(avg('true_positives_mean')))
    fp = int(round(avg('false_positives_mean')))
    tn = int(round(avg('true_negatives_mean')))
    fn = int(round(avg('false_negatives_mean')))

    return [
        'Overall (Avg)',
        str(tp), str(fp), str(tn), str(fn),
        _format_percent(avg('precision_mean')),
        _format_percent(avg('recall_mean')),
        _format_percent(avg('f1_mean')),
    ]


def _draw_table(
    rows: List[List[str]],
    columns: List[str],
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    ax.axis('off')

    # Title block
    ax.text(0.02, 0.98, title, fontsize=14, fontweight='bold', va='top')
    ax.text(0.02, 0.92, subtitle, fontsize=12, va='top')

    # Table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        colLoc='center',
        loc='center',
        bbox=[0.02, 0.1, 0.96, 0.75],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    # Give the first column more horizontal space for long method names.
    table.auto_set_column_width(col=[0])
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_width(cell.get_width() * 1.35)

    # Style header
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        if row == 0:
            cell.set_text_props(fontweight='bold')

    # Draw horizontal rules (booktabs style)
    ax.plot([0.02, 0.98], [0.82, 0.82], color='black', linewidth=1.2)
    ax.plot([0.02, 0.98], [0.77, 0.77], color='black', linewidth=0.8)
    ax.plot([0.02, 0.98], [0.1, 0.1], color='black', linewidth=1.2)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate a results table image from multi-seed JSON')
    parser.add_argument('--input', required=True, help='Path to multiseed_results_*.json')
    parser.add_argument('--output', required=True, help='Output PNG path')
    parser.add_argument('--title', default='Table 1.', help='Table title')
    parser.add_argument('--subtitle', default='Multi-seed detection results.', help='Table subtitle')
    parser.add_argument('--include-overall', action='store_true', help='Add an overall average row')
    parser.add_argument('--methods', default='', help='Comma-separated method list (optional)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = _load_results(input_path)
    aggregated = results['aggregated_results']

    if args.methods:
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    else:
        methods = sorted(aggregated.keys())

    rows = _build_rows(aggregated, methods)
    if args.include_overall:
        rows.append(_compute_overall(aggregated, methods))

    columns = ['Detection Method', 'TP', 'FP', 'TN', 'FN', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']

    _draw_table(rows, columns, args.title, args.subtitle, output_path)
    print(f"Saved table to: {output_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
