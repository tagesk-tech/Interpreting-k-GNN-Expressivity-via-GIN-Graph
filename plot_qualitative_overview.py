r"""Generate side-by-side visualisation figures for qualitative inspection.

For each dataset (MUTAG, PROTEINS), produces:
  - real_samples.png             : 5 real graphs per class (10 total)
  - random_samples_{model}.png   : 5 random generated samples per class
  - real_vs_random_{model}_c{N}.png : 5 random generated vs 5 real

Methodology note
----------------
We deliberately render *random* draws from the saved generated NPZ arrays
rather than the validation-score top-N. The generator maps a random latent
vector to one example at a time, so the validation-score top-N is a sample
from the right tail of per-draw randomness rather than a characterisation
of what the generator typically produces. The population-level evidence in
section 5.B (cycle counts, edge-pair matrices, node-type composition
tables) is computed from the entire 1000-sample population in
`aggregate_fingerprints.py` and `compare_datasets.py`; the figures here are
illustrative only and feed nothing back into the section 5.B quantitative
claims.

Inputs:
  - results/{dataset}/comparison/{model}_generated.npz  (1000 per class)
  - data/{DATASET}/                                     (real graphs via TUDataset)

Outputs (in results/{dataset}/comparison/figures/):
  - real_samples.png
  - random_samples_{model}.png        for model in {1gnn, 12gnn}
  - real_vs_random_{model}_class{N}.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from data_loader import load_dataset
from gin_handlers import get_handler


MAX_NODES = {"mutag": 28, "proteins": 50}
RANDOM_SEED = 0  # fixed so the figures are reproducible


def real_graphs_for_class(dataset_name: str, class_idx: int, n: int = 5, max_nodes: int = 50):
    """Return n (adj, x, num_nodes) tuples from the real dataset for a given class."""
    ds = load_dataset(dataset_name)
    picks = []
    for data in ds:
        if int(data.y.item()) != class_idx:
            continue
        num = int(data.num_nodes)
        if num > max_nodes:
            continue
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


def class_label(dataset_name: str, class_idx: int) -> str:
    """Return the canonical class label from the dataset handler."""
    handler = get_handler(dataset_name)
    return handler.class_names.get(class_idx, f"Class {class_idx}")


def load_random_generated(dataset_name: str, model: str, class_idx: int, k: int = 5,
                          rng: np.random.Generator | None = None):
    """Return k uniformly random generated graphs and their active-node counts."""
    npz = np.load(f"results/{dataset_name}/comparison/{model}_generated.npz")
    adjs = npz[f"adjs_class{class_idx}"]
    xs = npz[f"xs_class{class_idx}"]
    rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(adjs.shape[0], size=k, replace=False)
    chosen_adjs = adjs[idx]
    chosen_xs = xs[idx]
    n_nodes = []
    for i in range(k):
        a = (chosen_adjs[i] > 0.5).astype(np.float32)
        np.fill_diagonal(a, 0)
        n_nodes.append(int((a.sum(axis=1) > 0).sum()))
    return chosen_adjs, chosen_xs, n_nodes


def plot_real_samples(dataset_name: str, out_dir: Path):
    handler = get_handler(dataset_name)
    n_per = 5
    max_n = MAX_NODES[dataset_name]

    fig, axes = plt.subplots(2, n_per, figsize=(3 * n_per, 6))
    for c in (0, 1):
        samples = real_graphs_for_class(dataset_name, c, n_per, max_nodes=max_n)
        class_name = class_label(dataset_name, c)
        titles = [f"n={n}" for (_, _, n) in samples]
        for ax, (adj, x, _), title in zip(axes[c], samples, titles):
            handler.plot_explanation_graph(adj, x, ax=ax, title=title)
        axes[c, 0].set_ylabel(f"Class {c}\n({class_name})", fontsize=11)
    fig.suptitle(f"Real samples from {dataset_name.upper()} (one row per class)", fontsize=13)
    fig.tight_layout()
    out = out_dir / "real_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_random_overview(dataset_name: str, model: str, out_dir: Path):
    """2x5 grid: row per class, 5 random generated samples per class.

    Random rather than top-ranked: see module docstring.
    """
    handler = get_handler(dataset_name)
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for c in (0, 1):
        adjs, xs, n_nodes = load_random_generated(dataset_name, model, c, k=5, rng=rng)
        titles = [f"n={n}" for n in n_nodes]
        for i, (a, x, title) in enumerate(zip(adjs, xs, titles)):
            handler.plot_explanation_graph(a, x, ax=axes[c, i], title=title)
        class_name = class_label(dataset_name, c)
        axes[c, 0].set_ylabel(f"Class {c}\n({class_name})", fontsize=11)
    fig.suptitle(
        f"{dataset_name.upper()} — {model.upper()} 5 random generated graphs per class "
        f"(seed={RANDOM_SEED}, not top-ranked)",
        fontsize=13,
    )
    fig.tight_layout()
    out = out_dir / f"random_samples_{model}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_real_vs_random(dataset_name: str, model: str, class_idx: int, out_dir: Path):
    """5 random generated (top row) vs 5 real (bottom row) for one model/class.

    Random rather than top-ranked: see module docstring.
    """
    handler = get_handler(dataset_name)
    max_n = MAX_NODES[dataset_name]
    rng = np.random.default_rng(RANDOM_SEED + class_idx)

    adjs, xs, n_nodes = load_random_generated(dataset_name, model, class_idx, k=5, rng=rng)
    real = real_graphs_for_class(dataset_name, class_idx, 5, max_nodes=max_n)
    class_name = class_label(dataset_name, class_idx)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    gen_titles = [f"n={n}" for n in n_nodes]
    real_titles = [f"n={n}" for (_, _, n) in real]

    for i, (a, x, title) in enumerate(zip(adjs, xs, gen_titles)):
        handler.plot_explanation_graph(a, x, ax=axes[0, i], title=title)
    for i, ((a, x, _), title) in enumerate(zip(real, real_titles)):
        handler.plot_explanation_graph(a, x, ax=axes[1, i], title=title)

    axes[0, 0].set_ylabel(f"Random gen.\nby {model.upper()}", fontsize=11)
    axes[1, 0].set_ylabel("Real", fontsize=11)
    fig.suptitle(
        f"{dataset_name.upper()} — Class {class_idx} ({class_name}): "
        f"{model.upper()} random generated (top) vs real (bottom)",
        fontsize=13,
    )
    fig.tight_layout()
    out = out_dir / f"real_vs_random_{model}_class{class_idx}.png"
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
            plot_random_overview(dataset, model, out_dir)
            for c in (0, 1):
                plot_real_vs_random(dataset, model, c, out_dir)


if __name__ == "__main__":
    main()
