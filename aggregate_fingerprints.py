"""Population-level structural fingerprints for generated vs real graphs.

Motivation
----------
The reviewer of PR #8 pointed out that picking the top-N validation-ranked
generated graphs is not a sound way to characterise what a generator has
learned: the generator takes a random latent vector and produces one example
at a time, so the top-K rank is mostly the rank of the random draws.
Population-level fingerprints over a large sample (>= 1000 graphs) reveal
systematic differences and tolerate per-draw noise.

This script builds three fingerprints per (dataset, class, source) cell and
plots them for §5.B of the thesis:

  1. Cycle-count distribution (3-, 4-, 5-, 6-cycles per graph). Cycle
     structure is the canonical case where 2-WL distinguishes graphs that
     1-WL cannot, so it is the most theory-relevant aggregate.
  2. Edge-type co-occurrence matrix. For each undirected edge in each
     generated/real graph, count the unordered pair (type_i, type_j). The
     matrix is normalised to a probability distribution per source.
  3. Degree-conditioned-on-node-type histograms. For each node type, the
     mean degree over the population. Reveals whether each generator places
     the right node types in the right structural slots (e.g., terminal vs
     interior atoms).

Usage
-----
    python aggregate_fingerprints.py --dataset mutag
    python aggregate_fingerprints.py --dataset proteins

Reads:
    results/{dataset}/comparison/{model}_generated.npz  (1000 per class)
    data/{DATASET}/                                     (real via PyG TU)

Writes:
    results/{dataset}/comparison/figures/fingerprint_cycles.png
    results/{dataset}/comparison/figures/fingerprint_edge_pairs_{model}.png
    results/{dataset}/comparison/figures/fingerprint_degree_by_type.png
    results/{dataset}/comparison/fingerprints.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CYCLE_LENGTHS = (3, 4, 5, 6)


# ── Data loading ────────────────────────────────────────────────────────


def load_real_dense(dataset_name: str, class_idx: int, max_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load real graphs in the same dense format as generated ones.

    Loads via PyG TUDataset and pads adjacency / features to max_nodes.
    Skips graphs larger than max_nodes (consistent with what the generator
    can produce).
    """
    from data_loader import load_dataset

    ds = load_dataset(dataset_name)
    adjs = []
    xs = []
    for data in ds:
        if int(data.y.item()) != class_idx:
            continue
        n = int(data.num_nodes)
        if n > max_nodes:
            continue
        adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        ei = data.edge_index.numpy()
        if ei.size > 0:
            adj[ei[0], ei[1]] = 1.0
        x = np.zeros((max_nodes, data.x.shape[1]), dtype=np.float32)
        x[:n] = data.x.numpy()
        adjs.append(adj)
        xs.append(x)
    return np.stack(adjs), np.stack(xs)


def load_generated(dataset_name: str, model: str, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(f"results/{dataset_name}/comparison/{model}_generated.npz")
    return npz[f"adjs_class{class_idx}"], npz[f"xs_class{class_idx}"]


# ── Per-graph extraction ───────────────────────────────────────────────


def binarise(adjs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold + symmetrise + zero diagonal for a batch of adjacencies."""
    a = (adjs > threshold).astype(np.int8)
    a = np.maximum(a, a.transpose(0, 2, 1))
    for k in range(a.shape[0]):
        np.fill_diagonal(a[k], 0)
    return a


def active_mask(adj: np.ndarray) -> np.ndarray:
    """Boolean mask of nodes with degree >= 1 in this graph."""
    return adj.sum(axis=1) > 0


def count_cycles_at_length(adj: np.ndarray, length: int) -> int:
    """Number of distinct undirected simple cycles of exactly `length` nodes.

    Triangles (length 3) and 4-cycles use exact closed-form corrections.
    Lengths 5 and 6 use depth-first enumeration so the count is exact
    regardless of graph density; on the small graphs we compare
    (n <= 50, average degree ~ 4) this is fast enough to run on 1000+
    graphs per cell in seconds.
    """
    A = adj.astype(np.int64)
    if length == 3:
        # closed walks of length 3 = 6 * triangles (exact)
        trace = int(np.trace(A @ A @ A))
        return trace // 6
    if length == 4:
        # 8 * C4 = trace(A^4) - 2 * E - 2 * sum_i d_i*(d_i - 1) (exact)
        A2 = A @ A
        trace = int(np.trace(A2 @ A2))
        E = int(A.sum() // 2)
        deg = A.sum(axis=1)
        s = int((deg * (deg - 1)).sum())
        return max((trace - 2 * E - 2 * s), 0) // 8
    if length in (5, 6):
        return _enumerate_simple_cycles(adj, length)
    raise ValueError(f"unsupported cycle length {length}")


def _enumerate_simple_cycles(adj: np.ndarray, k: int) -> int:
    """Count simple cycles of exact length k by depth-first enumeration.

    For each starting vertex i, we explore paths of length k-1 to a
    neighbour of i, requiring all intermediate vertices to be greater
    than i (rooting at the smallest index avoids counting each cycle k
    times) and all distinct (avoids counting cycles with chords). Each
    cycle is then counted twice (one orientation per direction), so we
    divide by 2 at the end.
    """
    n = adj.shape[0]
    nbrs = [np.flatnonzero(adj[v]).tolist() for v in range(n)]
    total = 0

    def dfs(start: int, current: int, depth: int, visited_mask: int) -> None:
        nonlocal total
        if depth == k:
            if adj[current, start]:
                total += 1
            return
        for w in nbrs[current]:
            if w <= start:
                continue
            bit = 1 << w
            if visited_mask & bit:
                continue
            dfs(start, w, depth + 1, visited_mask | bit)

    for start in range(n):
        dfs(start, start, 1, 1 << start)
    return total // 2


def cycle_counts(adj: np.ndarray) -> Dict[int, int]:
    """Cycle count vector for one graph, restricted to active nodes."""
    mask = active_mask(adj)
    if mask.sum() < 3:
        return {k: 0 for k in CYCLE_LENGTHS}
    sub = adj[np.ix_(mask, mask)]
    return {k: count_cycles_at_length(sub, k) for k in CYCLE_LENGTHS}


def edge_pair_matrix(adj: np.ndarray, x: np.ndarray, num_types: int) -> np.ndarray:
    """Symmetric (T x T) count of (type_i, type_j) over undirected edges."""
    T = num_types
    out = np.zeros((T, T), dtype=np.int64)
    types = x.argmax(axis=1)
    iu, ju = np.triu_indices_from(adj, k=1)
    edge_mask = adj[iu, ju] > 0
    if edge_mask.sum() == 0:
        return out
    ti = types[iu[edge_mask]]
    tj = types[ju[edge_mask]]
    np.add.at(out, (ti, tj), 1)
    np.add.at(out, (tj, ti), 1)
    return out  # double-counted; we normalise below


def degree_by_type(adj: np.ndarray, x: np.ndarray, num_types: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sum of degrees per node type and node count per type for active nodes.

    Returns (deg_sum[T], count[T]) so the caller can aggregate across the
    population and compute means at the end.
    """
    deg = adj.sum(axis=1)
    mask = deg > 0
    if mask.sum() == 0:
        return np.zeros(num_types), np.zeros(num_types)
    types = x[mask].argmax(axis=1)
    deg = deg[mask]
    deg_sum = np.bincount(types, weights=deg, minlength=num_types)
    count = np.bincount(types, minlength=num_types)
    return deg_sum.astype(np.float64), count.astype(np.float64)


# ── Population aggregates ──────────────────────────────────────────────


def population_fingerprint(adjs: np.ndarray, xs: np.ndarray, num_types: int) -> Dict:
    """Aggregate the three fingerprints across a population of graphs."""
    a_bin = binarise(adjs)
    n = a_bin.shape[0]

    # Per-graph cycle counts
    cyc = {k: np.zeros(n, dtype=np.int64) for k in CYCLE_LENGTHS}
    for i in range(n):
        c = cycle_counts(a_bin[i])
        for k in CYCLE_LENGTHS:
            cyc[k][i] = c[k]

    # Edge-pair matrix summed over the population
    pair_total = np.zeros((num_types, num_types), dtype=np.int64)
    for i in range(n):
        pair_total += edge_pair_matrix(a_bin[i], xs[i], num_types)
    pair_norm = pair_total / max(int(pair_total.sum()), 1)

    # Degree-by-type means
    deg_sum_all = np.zeros(num_types, dtype=np.float64)
    count_all = np.zeros(num_types, dtype=np.float64)
    for i in range(n):
        ds, c = degree_by_type(a_bin[i], xs[i], num_types)
        deg_sum_all += ds
        count_all += c
    mean_deg_per_type = np.where(count_all > 0, deg_sum_all / np.maximum(count_all, 1), 0.0)
    type_population = count_all / max(count_all.sum(), 1)

    return {
        "cycle_counts": {k: cyc[k].tolist() for k in CYCLE_LENGTHS},
        "cycle_means": {k: float(cyc[k].mean()) for k in CYCLE_LENGTHS},
        "edge_pair_dist": pair_norm.tolist(),
        "edge_pair_count_total": int(pair_total.sum()),
        "mean_degree_by_type": mean_deg_per_type.tolist(),
        "type_population_share": type_population.tolist(),
    }


# ── Plotting ────────────────────────────────────────────────────────────


SOURCE_ORDER = ("real", "1gnn", "12gnn")
SOURCE_COLOUR = {
    "real": "#444444",
    "1gnn": "#1f77b4",
    "12gnn": "#d62728",
}
SOURCE_LABEL = {"real": "Real", "1gnn": "1-GNN gen", "12gnn": "1-2-GNN gen"}


def plot_cycle_fingerprint(fps: Dict, dataset_name: str, out_path: Path):
    """One row per class, one column per cycle length, bars for each source.

    Each bar shows the population mean number of cycles of that length per
    graph; error bars give the standard error of the mean over the
    population.
    """
    fig, axes = plt.subplots(2, len(CYCLE_LENGTHS), figsize=(3.4 * len(CYCLE_LENGTHS), 6.5),
                             sharey="row")
    for c in (0, 1):
        for j, length in enumerate(CYCLE_LENGTHS):
            ax = axes[c, j]
            means = []
            sems = []
            for src in SOURCE_ORDER:
                arr = np.array(fps[(c, src)]["cycle_counts"][length])
                means.append(float(arr.mean()))
                sems.append(float(arr.std(ddof=1) / np.sqrt(max(len(arr), 1))) if len(arr) > 1 else 0.0)
            colours = [SOURCE_COLOUR[s] for s in SOURCE_ORDER]
            ax.bar(range(len(SOURCE_ORDER)), means, yerr=sems, capsize=4,
                   color=colours, edgecolor="white")
            ax.set_xticks(range(len(SOURCE_ORDER)))
            ax.set_xticklabels([SOURCE_LABEL[s] for s in SOURCE_ORDER], rotation=20, fontsize=9)
            ax.set_title(f"{length}-cycles" if c == 0 else "")
            if j == 0:
                ax.set_ylabel(f"Class {c}\nmean per graph", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{dataset_name.upper()} cycle-count fingerprint "
                 "(mean per graph, error bars = SEM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_edge_pair_fingerprint(fps: Dict, dataset_name: str, type_labels: Dict[int, str],
                               out_dir: Path):
    """One figure per generator: heatmap of edge-type pair distribution
    side by side with the real distribution, and the absolute difference.
    """
    num_types = len(type_labels)
    labels = [type_labels.get(i, f"t{i}") for i in range(num_types)]
    for model in ("1gnn", "12gnn"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 9))
        for c in (0, 1):
            real = np.array(fps[(c, "real")]["edge_pair_dist"])
            gen = np.array(fps[(c, model)]["edge_pair_dist"])
            diff = gen - real
            vmax = max(real.max(), gen.max(), 1e-6)
            for ax, mat, title in zip(
                axes[c],
                (real, gen, diff),
                (f"Real (class {c})", f"{model.upper()} gen (class {c})", "gen − real"),
            ):
                if title.startswith("gen"):
                    cap = float(np.abs(mat).max() or 1e-6)
                    im = ax.imshow(mat, cmap="RdBu_r", vmin=-cap, vmax=cap)
                else:
                    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
                ax.set_xticks(range(num_types))
                ax.set_yticks(range(num_types))
                ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_title(title, fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{dataset_name.upper()} edge-type co-occurrence fingerprint — {model.upper()}",
            fontsize=12,
        )
        fig.tight_layout()
        out = out_dir / f"fingerprint_edge_pairs_{model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_degree_by_type(fps: Dict, dataset_name: str, type_labels: Dict[int, str],
                        out_path: Path):
    """One row per class, grouped bars per type, three sources per group."""
    num_types = len(type_labels)
    labels = [type_labels.get(i, f"t{i}") for i in range(num_types)]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.2 * num_types), 7), sharex=True)
    width = 0.27
    x = np.arange(num_types)
    for c in (0, 1):
        ax = axes[c]
        for k, src in enumerate(SOURCE_ORDER):
            mean_deg = np.array(fps[(c, src)]["mean_degree_by_type"])
            share = np.array(fps[(c, src)]["type_population_share"])
            # mute the bar when this type is essentially absent (< 0.5%)
            alpha = np.where(share < 0.005, 0.25, 1.0)
            offset = (k - 1) * width
            for i in range(num_types):
                ax.bar(x[i] + offset, mean_deg[i], width=width,
                       color=SOURCE_COLOUR[src], alpha=float(alpha[i]),
                       edgecolor="white",
                       label=SOURCE_LABEL[src] if (i == 0 and c == 0) else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f"Class {c}\nmean degree", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_xlabel("Node type")
    fig.suptitle(
        f"{dataset_name.upper()} degree conditioned on node type "
        "(faint bars: type < 0.5% of population)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Driver ─────────────────────────────────────────────────────────────


MAX_NODES = {"mutag": 28, "proteins": 50}


def build_fingerprints(dataset_name: str, num_types: int) -> Dict:
    fps: Dict[Tuple[int, str], Dict] = {}
    max_n = MAX_NODES[dataset_name]
    for c in (0, 1):
        adjs, xs = load_real_dense(dataset_name, c, max_n)
        fps[(c, "real")] = population_fingerprint(adjs, xs, num_types)
        print(f"  real class {c}: {adjs.shape[0]} graphs")
        for model in ("1gnn", "12gnn"):
            adjs, xs = load_generated(dataset_name, model, c)
            fps[(c, model)] = population_fingerprint(adjs, xs, num_types)
            print(f"  {model} class {c}: {adjs.shape[0]} graphs")
    return fps


def serialisable_summary(fps: Dict) -> Dict:
    """Strip per-graph cycle arrays from the JSON dump (means only)."""
    out = {}
    for (c, src), d in fps.items():
        out[f"class{c}_{src}"] = {
            "cycle_means": d["cycle_means"],
            "edge_pair_dist": d["edge_pair_dist"],
            "edge_pair_count_total": d["edge_pair_count_total"],
            "mean_degree_by_type": d["mean_degree_by_type"],
            "type_population_share": d["type_population_share"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mutag", "proteins"], required=True)
    args = ap.parse_args()

    from gin_handlers import get_handler

    handler = get_handler(args.dataset)
    num_types = len(handler.node_labels)
    type_labels = handler.node_labels

    out_dir = Path(f"results/{args.dataset}/comparison")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.dataset.upper()}] building population fingerprints...")
    fps = build_fingerprints(args.dataset, num_types)

    plot_cycle_fingerprint(fps, args.dataset, fig_dir / "fingerprint_cycles.png")
    plot_edge_pair_fingerprint(fps, args.dataset, type_labels, fig_dir)
    plot_degree_by_type(fps, args.dataset, type_labels, fig_dir / "fingerprint_degree_by_type.png")
    print(f"  wrote figures to {fig_dir}")

    summary_path = out_dir / "fingerprints.json"
    with open(summary_path, "w") as f:
        json.dump(serialisable_summary(fps), f, indent=2)
    print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
