"""Plot normalized cycle-count fingerprints from existing summaries.

This is a lightweight companion to aggregate_fingerprints.py for the thesis
figures: it uses the already committed cycle means plus size/degree summaries
to show whether raw cycle differences survive basic graph-size normalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CYCLE_LENGTHS = (3, 4, 5, 6)
SOURCE_ORDER = ("real", "1gnn", "12gnn")
SOURCE_COLOUR = {
    "real": "#444444",
    "1gnn": "#1f77b4",
    "12gnn": "#d62728",
}
SOURCE_LABEL = {"real": "Real", "1gnn": "1-GNN gen", "12gnn": "1-2-GNN gen"}


def load_normalized_values(dataset: str) -> Dict:
    base = Path("results") / dataset / "comparison"
    with open(base / "fingerprints.json") as f:
        fingerprints = json.load(f)
    with open(base / "report.json") as f:
        report = json.load(f)

    structural = report["structural_comparison"]
    out = {}
    for class_idx in (0, 1):
        real_stats = structural[f"1gnn_class{class_idx}"]
        source_stats = {
            "real": {
                "mean_size": real_stats["real_mean_size"],
                "mean_degree": real_stats["real_mean_degree"],
            },
            "1gnn": {
                "mean_size": structural[f"1gnn_class{class_idx}"]["gen_mean_size"],
                "mean_degree": structural[f"1gnn_class{class_idx}"]["gen_mean_degree"],
            },
            "12gnn": {
                "mean_size": structural[f"12gnn_class{class_idx}"]["gen_mean_size"],
                "mean_degree": structural[f"12gnn_class{class_idx}"]["gen_mean_degree"],
            },
        }

        for source in SOURCE_ORDER:
            cycles = {
                int(length): float(value)
                for length, value in fingerprints[f"class{class_idx}_{source}"]["cycle_means"].items()
            }
            mean_size = source_stats[source]["mean_size"]
            mean_edges = mean_size * source_stats[source]["mean_degree"] / 2.0
            out[(class_idx, source)] = {
                "per_node": {
                    length: value / mean_size if mean_size > 0 else 0.0
                    for length, value in cycles.items()
                },
                "per_edge": {
                    length: value / mean_edges if mean_edges > 0 else 0.0
                    for length, value in cycles.items()
                },
            }
    return out


def plot_dataset(dataset: str) -> Path:
    values = load_normalized_values(dataset)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    x = np.arange(len(CYCLE_LENGTHS))
    width = 0.24
    panels = (("per_node", "cycles per mean active node"), ("per_edge", "cycles per mean edge"))

    for class_idx in (0, 1):
        for col_idx, (key, title) in enumerate(panels):
            ax = axes[class_idx, col_idx]
            for offset, source in zip((-width, 0, width), SOURCE_ORDER):
                y = [values[(class_idx, source)][key][length] for length in CYCLE_LENGTHS]
                ax.bar(
                    x + offset,
                    y,
                    width=width,
                    color=SOURCE_COLOUR[source],
                    edgecolor="white",
                    label=SOURCE_LABEL[source] if class_idx == 0 and col_idx == 0 else None,
                )
            ax.set_title(f"Class {class_idx}: {title}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(length) for length in CYCLE_LENGTHS])
            ax.set_xlabel("Cycle length")
            ax.grid(axis="y", alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel("normalized mean", fontsize=10)

    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        f"{dataset.upper()} normalized cycle-count fingerprint "
        "(raw cycle means divided by mean graph size or mean edge count)",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = Path("results") / dataset / "comparison" / "figures" / "fingerprint_cycles_normalized.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mutag", "proteins", "all"], default="all")
    args = parser.parse_args()

    datasets = ("mutag", "proteins") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        print(f"wrote {plot_dataset(dataset)}")


if __name__ == "__main__":
    main()
