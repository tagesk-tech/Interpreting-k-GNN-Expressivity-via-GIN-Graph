"""Plot longer simple-cycle diagnostics for generated vs real graphs.

This is an appendix-style diagnostic for the population comparison. The main
thesis figure keeps 3--6 cycles because those are easier to read and include
the theory-relevant triangle signal. This script extends the exact simple-cycle
count to 9-cycles to check whether longer closures reveal an additional trend.

Reads:
    data/{MUTAG,PROTEINS}/raw/
    results/{dataset}/comparison/{model}_generated.npz

Writes:
    results/{dataset}/comparison/long_cycle_counts.json
    results/{dataset}/comparison/figures/fingerprint_cycles_3_to_9.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CYCLE_LENGTHS = tuple(range(3, 10))
MAX_NODES = {"mutag": 28, "proteins": 50}
RAW_NAMES = {"mutag": "MUTAG", "proteins": "PROTEINS"}
RAW_CLASS_MAP = {"mutag": {-1: 0, 1: 1}, "proteins": {1: 0, 2: 1}}
SOURCE_ORDER = ("real", "1gnn", "12gnn")
SOURCE_LABEL = {"real": "Real", "1gnn": "1-GNN gen", "12gnn": "1-2-GNN gen"}
SOURCE_COLOUR = {"real": "#444444", "1gnn": "#1f77b4", "12gnn": "#d62728"}


def count_simple_cycles(adj: List[Tuple[int, ...]], length: int) -> int:
    """Count undirected simple cycles of exactly `length` nodes."""
    total = 0

    def dfs(start: int, current: int, depth: int, visited_mask: int) -> None:
        nonlocal total
        if depth == length:
            if start in adj[current]:
                total += 1
            return
        for nxt in adj[current]:
            if nxt <= start:
                continue
            bit = 1 << nxt
            if visited_mask & bit:
                continue
            dfs(start, nxt, depth + 1, visited_mask | bit)

    for start in range(len(adj)):
        dfs(start, start, 1, 1 << start)
    return total // 2


def dense_to_adj_lists(adj: np.ndarray) -> List[Tuple[int, ...]]:
    """Threshold a dense adjacency and return active-node adjacency lists."""
    binary = (adj > 0.5).astype(np.int8)
    binary = np.maximum(binary, binary.T)
    np.fill_diagonal(binary, 0)
    active = np.flatnonzero(binary.sum(axis=1) > 0)
    if active.size == 0:
        return []
    sub = binary[np.ix_(active, active)]
    return [tuple(np.flatnonzero(row).tolist()) for row in sub]


def load_generated_graphs(dataset: str, model: str, class_idx: int) -> List[List[Tuple[int, ...]]]:
    npz = np.load(f"results/{dataset}/comparison/{model}_generated.npz")
    return [dense_to_adj_lists(adj) for adj in npz[f"adjs_class{class_idx}"]]


def load_real_graphs(dataset: str, class_idx: int, data_root: Path) -> List[List[Tuple[int, ...]]]:
    """Read TU raw files directly and skip graphs larger than the generator cap."""
    raw_name = RAW_NAMES[dataset]
    raw_dir = data_root / raw_name / "raw"
    graph_indicator = [
        int(value) for value in (raw_dir / f"{raw_name}_graph_indicator.txt").read_text().splitlines()
    ]
    raw_labels = [
        int(value) for value in (raw_dir / f"{raw_name}_graph_labels.txt").read_text().splitlines()
    ]

    graph_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graph_nodes[graph_id].append(node_id)

    edges_by_graph: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for line in (raw_dir / f"{raw_name}_A.txt").read_text().splitlines():
        left, right = line.replace(" ", "").split(",")
        u, v = int(left), int(right)
        graph_id = graph_indicator[u - 1]
        if graph_id != graph_indicator[v - 1]:
            continue
        a, b = sorted((u, v))
        if a != b:
            edges_by_graph[graph_id].add((a, b))

    graphs: List[List[Tuple[int, ...]]] = []
    for graph_id, nodes in graph_nodes.items():
        mapped_class = RAW_CLASS_MAP[dataset][raw_labels[graph_id - 1]]
        if mapped_class != class_idx or len(nodes) > MAX_NODES[dataset]:
            continue

        local_index = {node_id: idx for idx, node_id in enumerate(nodes)}
        adj = [set() for _ in nodes]
        for u, v in edges_by_graph[graph_id]:
            a, b = local_index[u], local_index[v]
            adj[a].add(b)
            adj[b].add(a)
        graphs.append([tuple(sorted(neighbours)) for neighbours in adj])
    return graphs


def cycle_summary(graphs: List[List[Tuple[int, ...]]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for length in CYCLE_LENGTHS:
        values = np.array([
            count_simple_cycles(graph, length) if len(graph) >= length else 0
            for graph in graphs
        ], dtype=np.float64)
        out[length] = {
            "mean": float(values.mean()),
            "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        }
    return out


def build_summary(dataset: str, data_root: Path) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    summary = {}
    for class_idx in (0, 1):
        class_key = f"class{class_idx}"
        summary[class_key] = {}
        for source in SOURCE_ORDER:
            if source == "real":
                graphs = load_real_graphs(dataset, class_idx, data_root)
            else:
                graphs = load_generated_graphs(dataset, source, class_idx)
            summary[class_key][source] = cycle_summary(graphs)
            print(f"{dataset} {class_key} {source}: {len(graphs)} graphs")
    return summary


def serialise_summary(summary: Dict) -> Dict:
    return {
        class_key: {
            source: {str(length): values for length, values in source_summary.items()}
            for source, source_summary in class_summary.items()
        }
        for class_key, class_summary in summary.items()
    }


def plot_summary(summary: Dict, dataset: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    x = np.array(CYCLE_LENGTHS)
    for class_idx, ax in enumerate(axes):
        class_summary = summary[f"class{class_idx}"]
        for source in SOURCE_ORDER:
            means = np.array([class_summary[source][length]["mean"] for length in CYCLE_LENGTHS])
            sems = np.array([class_summary[source][length]["sem"] for length in CYCLE_LENGTHS])
            ax.errorbar(
                x, means, yerr=sems, marker="o", linewidth=2, capsize=3,
                label=SOURCE_LABEL[source], color=SOURCE_COLOUR[source],
            )
        if dataset == "proteins":
            ax.set_yscale("log")
            ax.set_ylabel(f"Class {class_idx}\nmean per graph (log)")
        else:
            ax.set_ylabel(f"Class {class_idx}\nmean per graph")
        ax.set_title(f"Class {class_idx}", fontsize=10)
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, ncol=3)
    axes[1].set_xlabel("Simple-cycle length")
    fig.suptitle(f"{dataset.upper()} exact simple-cycle counts, lengths 3-9", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mutag", "proteins"), required=True)
    parser.add_argument("--data-root", default="data", type=Path)
    args = parser.parse_args()

    out_dir = Path(f"results/{args.dataset}/comparison")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args.dataset, args.data_root)
    summary_path = out_dir / "long_cycle_counts.json"
    summary_path.write_text(json.dumps(serialise_summary(summary), indent=2))
    plot_summary(summary, args.dataset, fig_dir / "fingerprint_cycles_3_to_9.png")
    print(f"wrote {summary_path}")
    print(f"wrote {fig_dir / 'fingerprint_cycles_3_to_9.png'}")


if __name__ == "__main__":
    main()
