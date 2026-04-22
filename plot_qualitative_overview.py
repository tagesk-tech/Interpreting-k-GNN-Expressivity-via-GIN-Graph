"""Generate consolidated qualitative-analysis figures.

For each dataset (MUTAG, PROTEINS), produces:
  - real_samples.png            : 5 real graphs per class (10 total)
  - overview_{model}.png        : 2x5 grid per model, row=class, cols=top-5 generated
  - real_vs_gen_{model}_c{N}.png: top-5 generated vs 5 real side by side

Uses only committed artefacts:
  - results/{dataset}/comparison/{model}_generated.npz  (500 generated per class)
  - results/{dataset}/{model}_class{N}/report.json      (top_explanations ranking)
  - data/{DATASET}/                                     (real graphs via TUDataset)

Saved into results/{dataset}/comparison/figures/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from data_loader import load_dataset
from gin_handlers import get_handler


MAX_NODES = {"mutag": 28, "proteins": 50}


def real_graphs_for_class(dataset_name: str, class_idx: int, n: int = 5, max_nodes: int = 50):
    """Return n (adj, x) tuples from the real dataset for a given class."""
    ds = load_dataset(dataset_name)
    picks = []
    for data in ds:
        if int(data.y.item()) != class_idx:
            continue
        num = int(data.num_nodes)
        if num > max_nodes:
            continue
        # build dense adj padded to max_nodes
        adj = torch.zeros(max_nodes, max_nodes)
        ei = data.edge_index
        if ei.numel() > 0:
            adj[ei[0], ei[1]] = 1.0
        x_pad = torch.zeros(max_nodes, data.x.shape[1])
        x_pad[:num] = data.x
        picks.append((adj.numpy(), x_pad.numpy(), num))
        if len(picks) >= n:
            break
    return picks


def plot_row(handler, graphs, axes, titles):
    for ax, (adj, x, n), title in zip(axes, graphs, titles):
        handler.plot_explanation_graph(adj, x, ax=ax, title=title)


def class_label_from_report(dataset_name: str, class_idx: int) -> str:
    """Read class_label from any existing report.json (consistent across reports).

    Authoritative over gin_handlers.class_names — those were changed by a later
    label-inversion fix but the on-disk report.json files were left as-is.
    """
    for model in ("1gnn", "12gnn"):
        path = Path(f"results/{dataset_name}/{model}_class{class_idx}/report.json")
        if path.exists():
            with open(path) as f:
                return json.load(f)["class_label"]
    return f"Class {class_idx}"


def plot_real_samples(dataset_name: str, out_dir: Path):
    handler = get_handler(dataset_name)
    n_per = 5
    max_n = MAX_NODES[dataset_name]

    fig, axes = plt.subplots(2, n_per, figsize=(3 * n_per, 6))
    for c in (0, 1):
        samples = real_graphs_for_class(dataset_name, c, n_per, max_nodes=max_n)
        class_name = class_label_from_report(dataset_name, c)
        titles = [f"n={n}" for (_, _, n) in samples]
        plot_row(handler, samples, axes[c], titles)
        axes[c, 0].set_ylabel(f"Class {c}\n({class_name})", fontsize=11)
    fig.suptitle(f"Real samples from {dataset_name.upper()} (one row per class)", fontsize=13)
    fig.tight_layout()
    out = out_dir / "real_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def load_top_generated(dataset_name: str, model: str, class_idx: int, top_k: int = 5):
    """Return top-K generated graphs using indices from report.json."""
    report_path = Path(f"results/{dataset_name}/{model}_class{class_idx}/report.json")
    with open(report_path) as f:
        report = json.load(f)
    top = report["top_explanations"][:top_k]
    indices = [t["index"] for t in top]
    scores = [t["validation_score"] for t in top]
    n_nodes = [t["num_nodes"] for t in top]

    # load the full NPZ to get adj/x for those indices
    # npz stores all generated graphs (50 or 100); `indices` in report.json
    # are into the original generation order, matching NPZ row order.
    npz_path = Path(f"results/{dataset_name}/{model}_class{class_idx}/explanations.npz")
    data = np.load(npz_path)
    adjs_all = data["adjs"]
    xs_all = data["xs"]
    npz_indices = data["indices"] if "indices" in data else np.arange(len(adjs_all))
    # map report-json indices -> npz row position
    pos = [int(np.where(npz_indices == i)[0][0]) for i in indices]
    adjs = adjs_all[pos]
    xs = xs_all[pos]
    return adjs, xs, scores, n_nodes


def plot_real_vs_gen(dataset_name: str, model: str, class_idx: int, out_dir: Path):
    """Top-5 generated (row 0) vs 5 real (row 1) for a given model/class."""
    handler = get_handler(dataset_name)
    max_n = MAX_NODES[dataset_name]

    adjs, xs, scores, n_nodes = load_top_generated(dataset_name, model, class_idx, top_k=5)
    real = real_graphs_for_class(dataset_name, class_idx, 5, max_nodes=max_n)
    class_name = class_label_from_report(dataset_name, class_idx)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    gen_titles = [f"v={s:.2f}, n={n}" for s, n in zip(scores, n_nodes)]
    real_titles = [f"n={n}" for (_, _, n) in real]

    for i, (a, x, title) in enumerate(zip(adjs, xs, gen_titles)):
        handler.plot_explanation_graph(a, x, ax=axes[0, i], title=title)
    for i, ((a, x, _), title) in enumerate(zip(real, real_titles)):
        handler.plot_explanation_graph(a, x, ax=axes[1, i], title=title)

    axes[0, 0].set_ylabel(f"Generated\nby {model.upper()}", fontsize=11)
    axes[1, 0].set_ylabel("Real", fontsize=11)
    fig.suptitle(
        f"{dataset_name.upper()} — Class {class_idx} ({class_name}): "
        f"{model.upper()} generated (top) vs real (bottom)",
        fontsize=13,
    )
    fig.tight_layout()
    out = out_dir / f"real_vs_gen_{model}_class{class_idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_model_overview(dataset_name: str, model: str, out_dir: Path):
    """2x5 grid: row per class, top-5 generated per class."""
    handler = get_handler(dataset_name)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for c in (0, 1):
        adjs, xs, scores, n_nodes = load_top_generated(dataset_name, model, c, top_k=5)
        titles = [f"v={s:.2f}, n={n}" for s, n in zip(scores, n_nodes)]
        for i, (a, x, title) in enumerate(zip(adjs, xs, titles)):
            handler.plot_explanation_graph(a, x, ax=axes[c, i], title=title)
        class_name = class_label_from_report(dataset_name, c)
        axes[c, 0].set_ylabel(f"Class {c}\n({class_name})", fontsize=11)
    fig.suptitle(
        f"{dataset_name.upper()} — {model.upper()} top-5 generated per class",
        fontsize=13,
    )
    fig.tight_layout()
    out = out_dir / f"overview_{model}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mutag", "proteins"])
    args = ap.parse_args()

    for dataset in args.datasets:
        out_dir = Path(f"results/{dataset}/comparison/figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{dataset.upper()}]")
        plot_real_samples(dataset, out_dir)
        for model in ("1gnn", "12gnn"):
            plot_model_overview(dataset, model, out_dir)
            for c in (0, 1):
                plot_real_vs_gen(dataset, model, c, out_dir)


if __name__ == "__main__":
    main()
