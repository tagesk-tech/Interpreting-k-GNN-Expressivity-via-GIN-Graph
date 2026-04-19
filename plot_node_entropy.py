#!/usr/bin/env python3
"""Plot graph-level normalized node-type entropy for real vs generated graphs.

This script treats each set of generated graphs as a synthetic class-conditional
dataset and compares its per-graph node-type entropy to the full real dataset
split by class.

Real-class entropies are computed directly from the raw TU files under
``data/{DATASET}/raw/`` so the script has no torch / torch_geometric dependency
and always reproduces the same numbers regardless of the PyG split logic.
The mapping from raw TU labels to PyG class indices follows ``torch.unique``
ascending order, matching PyG's ``TUDataset`` behaviour:

* MUTAG: raw label ``-1`` (63 non-mutagens) -> class 0 (Non-Mutagen);
  raw label ``+1`` (125 mutagens) -> class 1 (Mutagen).
* PROTEINS: raw label ``1`` (663 enzymes) -> class 0 (Enzyme);
  raw label ``2`` (450 non-enzymes) -> class 1 (Non-Enzyme).

Generated-class entropies use the saved ``.npz`` files from
``results/{DATASET}/comparison/{model}_generated.npz``.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Canonical class names. Kept here (rather than imported from config) so the
# script does not require torch or PyG to run.
CLASS_NAMES: Dict[str, Dict[int, str]] = {
    "mutag": {0: "Non-Mutagen", 1: "Mutagen"},
    "proteins": {0: "Enzyme", 1: "Non-Enzyme"},
}

NUM_NODE_TYPES: Dict[str, int] = {
    "mutag": 7,
    "proteins": 3,
}


def normalized_entropy(counts: np.ndarray, num_types: int) -> float:
    """Compute node-type entropy normalized to [0, 1]."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(num_types) if num_types > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def load_real_graph_entropies(dataset_name: str, data_root: str) -> Dict[int, np.ndarray]:
    """Compute per-graph normalized entropy for all raw TU graphs, grouped by
    PyG class index (ascending-sort remap from raw labels)."""
    raw_dir = os.path.join(data_root, dataset_name.upper(), "raw")
    prefix = dataset_name.upper()

    with open(os.path.join(raw_dir, f"{prefix}_graph_labels.txt")) as f:
        raw_labels = [int(line.strip()) for line in f if line.strip()]
    with open(os.path.join(raw_dir, f"{prefix}_graph_indicator.txt")) as f:
        graph_ids = [int(line.strip()) for line in f if line.strip()]
    with open(os.path.join(raw_dir, f"{prefix}_node_labels.txt")) as f:
        node_labels = [int(line.strip()) for line in f if line.strip()]

    unique_sorted = sorted(set(raw_labels))
    raw_to_class = {v: i for i, v in enumerate(unique_sorted)}

    per_graph_types: Dict[int, List[int]] = {}
    for gid, label in zip(graph_ids, node_labels):
        per_graph_types.setdefault(gid, []).append(label)

    num_types = NUM_NODE_TYPES[dataset_name]
    class_entropies: Dict[int, List[float]] = {0: [], 1: []}
    for idx, raw_label in enumerate(raw_labels):
        gid = idx + 1
        cls = raw_to_class[raw_label]
        counts = np.bincount(per_graph_types[gid], minlength=num_types)
        class_entropies[cls].append(normalized_entropy(counts, num_types))

    return {c: np.asarray(v) for c, v in class_entropies.items()}


def generated_graph_entropies(adjs: np.ndarray, xs: np.ndarray) -> np.ndarray:
    num_types = xs.shape[-1]
    values: List[float] = []
    for adj, x in zip(adjs, xs):
        adj_bin = (adj > 0.5).astype(float)
        np.fill_diagonal(adj_bin, 0)
        active = adj_bin.sum(axis=1) > 0
        if not active.any():
            continue
        type_ids = x[active].argmax(axis=1)
        counts = np.bincount(type_ids, minlength=num_types)
        values.append(normalized_entropy(counts, num_types))
    return np.asarray(values)


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "n": int(values.size),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot normalized node-type entropy for real vs generated graphs"
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing raw TU dataset folders",
    )
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="Directory containing generated comparison datasets",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/summary",
        help="Directory to save the figure and summary JSON",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["mutag", "proteins"]
    models = ["1gnn", "12gnn"]

    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for row, dataset_name in enumerate(datasets):
        real_by_class = load_real_graph_entropies(dataset_name, args.data_dir)
        summary[dataset_name] = {}

        for col, class_idx in enumerate([0, 1]):
            ax = axes[row, col]
            class_name = CLASS_NAMES[dataset_name][class_idx]
            real_values = real_by_class[class_idx]

            panel_data = [real_values]
            panel_labels = ["Real", "1-GNN", "1-2-GNN"]

            summary[dataset_name][str(class_idx)] = {
                "class_name": class_name,
                "real": summarize(real_values),
            }

            for model_name in models:
                generated_path = os.path.join(
                    args.results_dir,
                    dataset_name,
                    "comparison",
                    f"{model_name}_generated.npz",
                )
                generated = np.load(generated_path)
                gen_values = generated_graph_entropies(
                    generated[f"adjs_class{class_idx}"],
                    generated[f"xs_class{class_idx}"],
                )
                panel_data.append(gen_values)
                summary[dataset_name][str(class_idx)][model_name] = summarize(gen_values)

            bp = ax.boxplot(
                panel_data,
                tick_labels=panel_labels,
                patch_artist=True,
                widths=0.65,
                showfliers=False,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.4)

            ax.set_title(f"{dataset_name.upper()} {class_name}")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Normalized Node-Type Entropy")
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Graph-Level Node-Type Entropy for Real and Generated Datasets",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(args.output_dir, "normalized_node_entropy.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(args.output_dir, "normalized_node_entropy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved figure to {fig_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
