#!/usr/bin/env python3
"""Create real-vs-generated example graph panels for the thesis.

The script is intentionally lightweight: it parses TU raw dataset files
directly and reads the saved top explanation arrays from results/. It avoids
PyTorch/PyG so the illustrative figures can be regenerated without the full
training environment.

Outputs:
  results/{dataset}/comparison/figures/example_graphs.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_name: str
    class_names: dict[int, str]
    node_labels: dict[int, str]
    node_colors: dict[str, str]
    max_nodes: int


DATASETS = {
    "mutag": DatasetSpec(
        name="MUTAG",
        raw_name="MUTAG",
        class_names={0: "Non-Mutagen", 1: "Mutagen"},
        node_labels={0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"},
        node_colors={
            "C": "#FFA500",
            "N": "#00BFFF",
            "O": "#FF0000",
            "F": "#32CD32",
            "I": "#800080",
            "Cl": "#90EE90",
            "Br": "#8B4513",
            "?": "#808080",
        },
        max_nodes=28,
    ),
    "proteins": DatasetSpec(
        name="PROTEINS",
        raw_name="PROTEINS",
        class_names={0: "Enzyme", 1: "Non-Enzyme"},
        node_labels={0: "H", 1: "S", 2: "C"},
        node_colors={
            "H": "#FF6B6B",
            "S": "#4ECDC4",
            "C": "#95E1D3",
            "?": "#808080",
        },
        max_nodes=50,
    ),
}


def read_int_rows(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(part.strip()) for part in line.replace(",", " ").split()])
    return rows


def load_real_graphs(dataset: str, data_root: Path) -> dict[int, list[dict]]:
    spec = DATASETS[dataset]
    raw_dir = data_root / spec.raw_name / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Could not find TU raw dataset directory: {raw_dir}")

    graph_indicator = [row[0] for row in read_int_rows(raw_dir / f"{spec.raw_name}_graph_indicator.txt")]
    graph_labels_raw = [row[0] for row in read_int_rows(raw_dir / f"{spec.raw_name}_graph_labels.txt")]
    node_labels = [row[0] for row in read_int_rows(raw_dir / f"{spec.raw_name}_node_labels.txt")]
    edge_rows = read_int_rows(raw_dir / f"{spec.raw_name}_A.txt")

    raw_to_class = {raw: idx for idx, raw in enumerate(sorted(set(graph_labels_raw)))}
    graph_count = len(graph_labels_raw)
    global_to_graph = {node_idx + 1: graph_id for node_idx, graph_id in enumerate(graph_indicator)}

    nodes_by_graph = [[] for _ in range(graph_count + 1)]
    for global_node, graph_id in enumerate(graph_indicator, start=1):
        nodes_by_graph[graph_id].append(global_node)

    local_index = {}
    for graph_id in range(1, graph_count + 1):
        for local_node, global_node in enumerate(nodes_by_graph[graph_id]):
            local_index[global_node] = local_node

    edges_by_graph = [set() for _ in range(graph_count + 1)]
    for u, v in edge_rows:
        graph_id = global_to_graph.get(u)
        if graph_id is None or graph_id != global_to_graph.get(v):
            continue
        a, b = local_index[u], local_index[v]
        if a == b:
            continue
        edges_by_graph[graph_id].add(tuple(sorted((a, b))))

    by_class = {0: [], 1: []}
    for graph_id in range(1, graph_count + 1):
        global_nodes = nodes_by_graph[graph_id]
        labels = [node_labels[node - 1] for node in global_nodes]
        class_idx = raw_to_class[graph_labels_raw[graph_id - 1]]
        by_class[class_idx].append({"labels": labels, "edges": sorted(edges_by_graph[graph_id])})

    return by_class


def active_generated_graph(adj: np.ndarray, x: np.ndarray) -> dict:
    edges_matrix = (adj > 0.5).astype(np.float32)
    np.fill_diagonal(edges_matrix, 0)
    degrees = edges_matrix.sum(axis=1)
    active = np.where(degrees > 0)[0]

    if len(active) == 0:
        return {"labels": [], "edges": []}

    index = {old: new for new, old in enumerate(active)}
    labels = np.argmax(x[active], axis=1).astype(int).tolist()
    edges = []
    for i_pos, i in enumerate(active):
        for j in active[i_pos + 1 :]:
            if edges_matrix[i, j] > 0:
                edges.append((index[i], index[j]))
    return {"labels": labels, "edges": edges}


def load_top_generated(dataset: str, model: str, class_idx: int, count: int) -> list[dict]:
    path = Path("results") / dataset / f"{model}_class{class_idx}" / "explanations.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved explanation file: {path}")
    data = np.load(path)
    adjs = data["adjs"][:count]
    xs = data["xs"][:count]
    return [active_generated_graph(adj, x) for adj, x in zip(adjs, xs)]


def choose_real_examples(real_by_class: dict[int, list[dict]], class_idx: int, count: int, max_nodes: int) -> list[dict]:
    candidates = [g for g in real_by_class[class_idx] if len(g["labels"]) <= max_nodes]
    if len(candidates) < count:
        candidates = real_by_class[class_idx]
    return candidates[:count]


def force_layout(node_count: int, edges: list[tuple[int, int]], seed: int) -> np.ndarray:
    if node_count == 0:
        return np.zeros((0, 2))
    if node_count == 1:
        return np.array([[0.0, 0.0]])

    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, node_count, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    pos += rng.normal(scale=0.04, size=pos.shape)

    area = 4.0
    k = np.sqrt(area / node_count)
    temperature = 0.18

    for _ in range(90):
        disp = np.zeros_like(pos)

        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.linalg.norm(delta, axis=2) + 1e-6
        np.fill_diagonal(distance, np.inf)
        repulsion = (k * k / distance)[:, :, None] * delta / distance[:, :, None]
        disp += np.nansum(repulsion, axis=1)

        for u, v in edges:
            delta_uv = pos[u] - pos[v]
            dist = float(np.linalg.norm(delta_uv) + 1e-6)
            attraction = (dist * dist / k) * (delta_uv / dist)
            disp[u] -= attraction
            disp[v] += attraction

        lengths = np.linalg.norm(disp, axis=1)
        limited = np.minimum(lengths, temperature)
        nonzero = lengths > 1e-9
        pos[nonzero] += disp[nonzero] / lengths[nonzero, None] * limited[nonzero, None]
        temperature *= 0.95

    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    if scale > 0:
        pos /= scale
    return pos


def draw_graph(ax: plt.Axes, graph: dict, spec: DatasetSpec, title: str, seed: int) -> None:
    labels_idx = graph["labels"]
    edges = graph["edges"]
    node_count = len(labels_idx)
    ax.axis("off")
    ax.set_title(title, fontsize=8, pad=2)

    if node_count == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        return

    pos = force_layout(node_count, edges, seed)

    for u, v in edges:
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], color="#555555", linewidth=0.7, alpha=0.65, zorder=1)

    labels = [spec.node_labels.get(int(idx), "?") for idx in labels_idx]
    colors = [spec.node_colors.get(label, spec.node_colors["?"]) for label in labels]
    node_size = 150 if node_count <= 30 else 70
    font_size = 6 if node_count <= 30 else 4

    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=colors, edgecolors="#222222", linewidths=0.4, zorder=2)
    for (x_coord, y_coord), label in zip(pos, labels):
        ax.text(x_coord, y_coord, label, ha="center", va="center", fontsize=font_size, fontweight="bold", zorder=3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)


def plot_dataset(dataset: str, data_root: Path, count: int) -> Path:
    spec = DATASETS[dataset]
    real_by_class = load_real_graphs(dataset, data_root)

    rows = [
        ("Real", lambda class_idx: choose_real_examples(real_by_class, class_idx, count, spec.max_nodes)),
        ("1-GNN gen.", lambda class_idx: load_top_generated(dataset, "1gnn", class_idx, count)),
        ("1-2-GNN gen.", lambda class_idx: load_top_generated(dataset, "12gnn", class_idx, count)),
    ]

    ncols = count * 2
    fig, axes = plt.subplots(len(rows), ncols, figsize=(2.35 * ncols, 2.25 * len(rows)))
    for row_idx, (row_label, graph_loader) in enumerate(rows):
        for class_idx in (0, 1):
            graphs = graph_loader(class_idx)
            offset = class_idx * count
            for col_idx in range(count):
                graph = graphs[col_idx]
                edge_count = len(graph["edges"])
                node_count = len(graph["labels"])
                title = f"n={node_count}, e={edge_count}"
                draw_graph(axes[row_idx, offset + col_idx], graph, spec, title, seed=101 + 41 * row_idx + 11 * class_idx + col_idx)
        axes[row_idx, 0].text(-0.12, 0.5, row_label, transform=axes[row_idx, 0].transAxes, ha="right", va="center", fontsize=8)

    fig.suptitle(f"{spec.name}: real examples and top generated explanations", fontsize=12)
    fig.text(0.34, 0.91, f"Class 0: {spec.class_names[0]}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(0.72, 0.91, f"Class 1: {spec.class_names[1]}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0.10, 0.0, 1.0, 0.90))

    out = Path("results") / dataset / "comparison" / "figures" / "example_graphs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=["mutag", "proteins"], choices=sorted(DATASETS))
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    for dataset in args.datasets:
        out = plot_dataset(dataset, args.data_root, args.count)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
