#!/usr/bin/env python3
"""Plot graph-level normalized node-type entropy for real vs generated graphs.

This script treats each set of generated graphs as a synthetic class-conditional
dataset and compares its per-graph node-type entropy to the corresponding real
training subset.
"""

import argparse
import json
import math
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DataConfig, get_class_name
from data_loader import create_data_loaders, load_dataset


def normalized_entropy(type_ids: np.ndarray, num_types: int) -> float:
    """Compute node-type entropy normalized to [0, 1]."""
    if type_ids.size == 0:
        return 0.0

    counts = np.bincount(type_ids, minlength=num_types).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(num_types) if num_types > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def real_graph_entropies(dataset_subset, num_types: int) -> np.ndarray:
    values: List[float] = []
    for data in dataset_subset:
        type_ids = data.x.argmax(dim=1).cpu().numpy()
        values.append(normalized_entropy(type_ids, num_types))
    return np.array(values)


def generated_graph_entropies(adjs: np.ndarray, xs: np.ndarray) -> np.ndarray:
    num_types = xs.shape[-1]
    values: List[float] = []
    for adj, x in zip(adjs, xs):
        adj_bin = (adj > 0.5).astype(float)
        np.fill_diagonal(adj_bin, 0)
        active = adj_bin.sum(axis=1) > 0
        type_ids = x[active].argmax(axis=1)
        values.append(normalized_entropy(type_ids, num_types))
    return np.array(values)


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
        dataset = load_dataset(dataset_name)
        data_config = DataConfig.from_dataset(dataset_name)
        _, _, train_dataset, _ = create_data_loaders(dataset, seed=data_config.seed)
        num_types = dataset.num_node_features
        summary[dataset_name] = {}

        for col, class_idx in enumerate([0, 1]):
            ax = axes[row, col]
            class_name = get_class_name(class_idx, dataset_name)
            real_subset = [d for d in train_dataset if d.y.item() == class_idx]
            real_values = real_graph_entropies(real_subset, num_types)

            panel_data = [real_values]
            panel_labels = ["Real", "1-GNN", "1-2-GNN"]

            summary[dataset_name][str(class_idx)] = {
                "class_name": class_name,
                "real": summarize(real_values),
            }

            for model_name in models:
                gen_path = os.path.join(
                    args.results_dir,
                    dataset_name,
                    "comparison",
                    f"{model_name}_generated.npz",
                )
                generated = np.load(gen_path)
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
