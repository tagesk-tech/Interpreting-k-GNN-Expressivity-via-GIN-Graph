#!/usr/bin/env python3
"""Compare GIN-Graph generated datasets against real MUTAG/PROTEINS data.

Usage:
    python compare_datasets.py --dataset mutag --num_samples 500
    python compare_datasets.py --dataset proteins --num_samples 500

Generates 4 synthetic datasets (2 models x 2 classes), then runs:
  A) Structural property comparison (degree, node types, graph size)
  B) Cross-model classification (classify generated graphs with both models)
  C) Embedding space visualization (t-SNE of real vs generated)

Outputs go to results/{dataset}/comparison/
"""

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.utils import to_dense_adj

from config import DataConfig, GINGraphConfig, get_class_name
from data_loader import (
    AVAILABLE_DATASETS,
    create_data_loaders,
    get_class_statistics,
    get_class_subset,
    load_dataset,
)
from gin_handlers import get_handler
from model_wrapper import DenseToSparseWrapper
from train_gin_graph import GINGraphTrainer, load_pretrained_kgnn

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False


# ── Helpers ──────────────────────────────────────────────────────────


def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def load_all_models(dataset_name, checkpoint_dir, gin_checkpoint_dir, device):
    """Load both k-GNN models and all 4 GIN-Graph trainers for one dataset.

    Returns:
        kgnns: {'1gnn': nn.Module, '12gnn': nn.Module}
        trainers: {('1gnn', 0): GINGraphTrainer, ...}
        train_dataset: the training split
        data_config: DataConfig
        class_stats: per-class statistics dict
    """
    dataset = load_dataset(dataset_name)
    data_config = DataConfig.from_dataset(dataset_name)
    _, _, train_dataset, _ = create_data_loaders(dataset, seed=data_config.seed)

    class_stats = get_class_statistics(train_dataset)
    for label in [0, 1]:
        nodes = [d.num_nodes for d in train_dataset if d.y.item() == label]
        class_stats[label]['avg_nodes'] = np.mean(nodes) if nodes else 0

    kgnns = {}
    trainers = {}

    for model_name in ['1gnn', '12gnn']:
        kgnn = load_pretrained_kgnn(model_name, checkpoint_dir, device, dataset_name)
        kgnns[model_name] = kgnn

        for target_class in [0, 1]:
            gin_path = os.path.join(
                gin_checkpoint_dir, dataset_name,
                f'{model_name}_class{target_class}.pt'
            )
            if not os.path.exists(gin_path):
                raise FileNotFoundError(f"GIN checkpoint not found: {gin_path}")

            gin_ckpt = torch.load(gin_path, map_location=device, weights_only=False)
            gin_config = gin_ckpt.get('config', GINGraphConfig())

            trainer = GINGraphTrainer(
                pretrained_gnn=kgnn,
                model_type=model_name,
                target_class=target_class,
                config=gin_config,
                data_config=data_config,
                device=device,
                class_stats=class_stats,
                dataset_name=dataset_name,
            )
            trainer.load_checkpoint(gin_path)

            if trainer.class_centroid is None:
                target_ds = get_class_subset(train_dataset, target_class)
                trainer.compute_class_centroid(target_ds)

            trainers[(model_name, target_class)] = trainer
            print(f"  Loaded {model_name} class {target_class} "
                  f"(epoch {trainer.epoch}, {trainer.global_step} steps)")

    return kgnns, trainers, train_dataset, data_config, class_stats


def generate_all(trainers, num_samples):
    """Generate graphs from all 4 trainers.

    Returns:
        generated: {(model, class): {'adjs': np.ndarray, 'xs': np.ndarray,
                                      'metrics': list}}
    """
    generated = {}
    for (model_name, target_class), trainer in trainers.items():
        class_label = get_class_name(target_class, trainer.dataset_name)
        print(f"  Generating {num_samples} graphs: {model_name} class {target_class} "
              f"({class_label})...")
        adjs, xs, metrics = trainer.generate_explanations(num_samples=num_samples)
        valid = sum(1 for m in metrics if m.is_valid)
        print(f"    → {valid}/{num_samples} valid "
              f"({valid/num_samples:.0%}), "
              f"mean v={np.mean([m.validation_score for m in metrics]):.3f}")
        generated[(model_name, target_class)] = {
            'adjs': adjs, 'xs': xs, 'metrics': metrics,
        }
    return generated


# ── Real graph extraction ────────────────────────────────────────────


def extract_real_properties(dataset_subset, max_nodes=None):
    """Extract structural properties from real PyG graphs.

    Returns dict with 'degrees', 'node_types', 'sizes', 'avg_degrees'.
    """
    degrees = []       # per-node degrees across all graphs
    node_types = []    # per-node type indices across all graphs
    sizes = []         # number of nodes per graph
    avg_degrees = []   # average degree per graph

    for data in dataset_subset:
        n = data.num_nodes
        e = data.num_edges  # directed count in PyG
        sizes.append(n)

        # Degree per node
        if data.edge_index.size(1) > 0:
            deg = torch.zeros(n, dtype=torch.long)
            deg.scatter_add_(0, data.edge_index[0],
                             torch.ones(data.edge_index.size(1), dtype=torch.long))
            degrees.extend(deg.tolist())
            avg_degrees.append(e / n)  # undirected: PyG stores both directions
        else:
            degrees.extend([0] * n)
            avg_degrees.append(0.0)

        # Node types (argmax of one-hot features)
        if data.x is not None:
            types = data.x.argmax(dim=1).tolist()
            node_types.extend(types)

    return {
        'degrees': np.array(degrees),
        'node_types': np.array(node_types),
        'sizes': np.array(sizes),
        'avg_degrees': np.array(avg_degrees),
    }


def extract_generated_properties(adjs, xs, edge_threshold=0.5):
    """Extract structural properties from generated dense graphs.

    Args:
        adjs: [N_samples, N, N] adjacency matrices
        xs: [N_samples, N, D] node features
    """
    degrees = []
    node_types = []
    sizes = []
    avg_degrees = []

    for i in range(adjs.shape[0]):
        adj = (adjs[i] > edge_threshold).astype(float)
        np.fill_diagonal(adj, 0)  # no self-loops
        x = xs[i]

        # Active nodes (non-isolated)
        node_deg = adj.sum(axis=1)
        active = node_deg > 0
        n_active = int(active.sum())
        n_edges = int(adj.sum()) // 2  # undirected

        sizes.append(n_active)
        if n_active > 0:
            degrees.extend(node_deg[active].astype(int).tolist())
            avg_degrees.append(2 * n_edges / n_active)
        else:
            avg_degrees.append(0.0)

        # Node types for active nodes
        types = x[active].argmax(axis=1)
        node_types.extend(types.tolist())

    return {
        'degrees': np.array(degrees),
        'node_types': np.array(node_types),
        'sizes': np.array(sizes),
        'avg_degrees': np.array(avg_degrees),
    }


# ── Analysis A: Structural comparison ────────────────────────────────


def plot_structural_comparison(real_props, gen_props, model_name, class_idx,
                               class_label, handler, save_path):
    """Create 2x2 structural comparison figure for one (model, class)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name.upper()} Class {class_idx} ({class_label}): '
                 f'Generated vs Real', fontsize=14, fontweight='bold')

    num_features = len(handler.node_labels)

    # (a) Degree distribution
    ax = axes[0, 0]
    max_deg = max(real_props['degrees'].max() if len(real_props['degrees']) else 0,
                  gen_props['degrees'].max() if len(gen_props['degrees']) else 0)
    bins = np.arange(0, max_deg + 2) - 0.5
    ax.hist(real_props['degrees'], bins=bins, density=True, alpha=0.5,
            color='steelblue', label='Real', edgecolor='white')
    ax.hist(gen_props['degrees'], bins=bins, density=True, alpha=0.5,
            color='indianred', label='Generated', edgecolor='white')
    if len(real_props['degrees']) > 0 and len(gen_props['degrees']) > 0:
        ks_stat, ks_p = sp_stats.ks_2samp(real_props['degrees'],
                                           gen_props['degrees'])
        ax.text(0.97, 0.97, f'KS={ks_stat:.3f}\np={ks_p:.3g}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Density')
    ax.set_title('Degree Distribution')
    ax.legend(fontsize=9)

    # (b) Node type distribution
    ax = axes[0, 1]
    real_counts = np.bincount(real_props['node_types'], minlength=num_features)
    gen_counts = np.bincount(gen_props['node_types'], minlength=num_features)
    real_frac = real_counts / max(real_counts.sum(), 1)
    gen_frac = gen_counts / max(gen_counts.sum(), 1)

    x_pos = np.arange(num_features)
    width = 0.35
    labels = [handler.node_labels.get(i, '?') for i in range(num_features)]
    ax.bar(x_pos - width/2, real_frac, width, color='steelblue',
           label='Real', edgecolor='white')
    ax.bar(x_pos + width/2, gen_frac, width, color='indianred',
           label='Generated', edgecolor='white')

    # Chi-square test
    if gen_counts.sum() > 0 and real_counts.sum() > 0:
        # Scale expected to match observed total
        expected = real_frac * gen_counts.sum()
        # Replace zeros with small value, then rescale to match sum
        expected = np.maximum(expected, 0.5)
        expected = expected * (gen_counts.sum() / expected.sum())
        chi2, chi2_p = sp_stats.chisquare(gen_counts, f_exp=expected)
        ax.text(0.97, 0.97, f'χ²={chi2:.1f}\np={chi2_p:.3g}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Proportion')
    ax.set_title('Node Type Distribution')
    ax.legend(fontsize=9)

    # (c) Graph size distribution
    ax = axes[1, 0]
    all_sizes = np.concatenate([real_props['sizes'], gen_props['sizes']])
    bins_size = np.arange(all_sizes.min(), all_sizes.max() + 2) - 0.5
    ax.hist(real_props['sizes'], bins=bins_size, density=True, alpha=0.5,
            color='steelblue', label='Real', edgecolor='white')
    ax.hist(gen_props['sizes'], bins=bins_size, density=True, alpha=0.5,
            color='indianred', label='Generated', edgecolor='white')
    if len(real_props['sizes']) > 0 and len(gen_props['sizes']) > 0:
        ks_s, ks_sp = sp_stats.ks_2samp(real_props['sizes'].astype(float),
                                         gen_props['sizes'].astype(float))
        ax.text(0.97, 0.97, f'KS={ks_s:.3f}\np={ks_sp:.3g}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Number of Active Nodes')
    ax.set_ylabel('Density')
    ax.set_title('Graph Size Distribution')
    ax.legend(fontsize=9)

    # (d) Average degree per graph (box plot)
    ax = axes[1, 1]
    bp = ax.boxplot([real_props['avg_degrees'], gen_props['avg_degrees']],
                    tick_labels=['Real', 'Generated'], patch_artist=True,
                    widths=0.5)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('indianred')
    bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel('Average Degree')
    ax.set_title('Average Degree per Graph')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_structural_analysis(generated, train_dataset, handler, data_config,
                            save_dir):
    """Run structural comparison for all (model, class) combos."""
    results = {}
    num_features = len(handler.node_labels)

    for (model_name, class_idx), data in generated.items():
        class_label = get_class_name(class_idx, handler.name.lower())

        # Real graphs for this class
        real_subset = get_class_subset(train_dataset, class_idx)
        real_props = extract_real_properties(real_subset)
        gen_props = extract_generated_properties(data['adjs'], data['xs'])

        # Statistical tests
        ks_deg = sp_stats.ks_2samp(real_props['degrees'], gen_props['degrees']) \
            if len(real_props['degrees']) > 0 and len(gen_props['degrees']) > 0 \
            else (0, 1)
        ks_size = sp_stats.ks_2samp(real_props['sizes'].astype(float),
                                    gen_props['sizes'].astype(float)) \
            if len(real_props['sizes']) > 0 and len(gen_props['sizes']) > 0 \
            else (0, 1)

        real_type_counts = np.bincount(real_props['node_types'],
                                       minlength=num_features)
        gen_type_counts = np.bincount(gen_props['node_types'],
                                      minlength=num_features)

        results[(model_name, class_idx)] = {
            'degree_ks_stat': float(ks_deg[0]),
            'degree_ks_pvalue': float(ks_deg[1]),
            'size_ks_stat': float(ks_size[0]),
            'size_ks_pvalue': float(ks_size[1]),
            'real_mean_degree': float(real_props['avg_degrees'].mean()),
            'gen_mean_degree': float(gen_props['avg_degrees'].mean()),
            'real_mean_size': float(real_props['sizes'].mean()),
            'gen_mean_size': float(gen_props['sizes'].mean()),
            'real_node_type_dist': {handler.node_labels.get(i, '?'):
                                    float(real_type_counts[i] / max(real_type_counts.sum(), 1))
                                    for i in range(num_features)},
            'gen_node_type_dist': {handler.node_labels.get(i, '?'):
                                   float(gen_type_counts[i] / max(gen_type_counts.sum(), 1))
                                   for i in range(num_features)},
        }

        # Plot
        fig_path = os.path.join(save_dir, f'structural_{model_name}_class{class_idx}.png')
        plot_structural_comparison(real_props, gen_props, model_name, class_idx,
                                  class_label, handler, fig_path)
        print(f"    Saved {fig_path}")

    return results


# ── Analysis B: Cross-model classification ───────────────────────────


def classify_batch(wrapper, adjs, xs, device, batch_size=64):
    """Classify dense graphs using a DenseToSparseWrapper.

    Returns predicted classes and class probabilities.
    """
    all_preds = []
    all_probs = []
    n = adjs.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_b = torch.tensor(adjs[start:end], dtype=torch.float32, device=device)
        a_b = torch.tensor(xs[start:end], dtype=torch.float32, device=device)
        # Note: wrapper.forward expects (x, adj) — features first
        # adjs[i] is adjacency, xs[i] is features
        x_feat = torch.tensor(xs[start:end], dtype=torch.float32, device=device)
        a_adj = torch.tensor(adjs[start:end], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = wrapper(x_feat, a_adj)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def run_cross_classification(generated, kgnns, data_config, device, handler,
                             save_dir):
    """Classify all generated graphs with both models."""
    # Create wrappers for classification
    wrappers = {}
    for model_name, kgnn in kgnns.items():
        wrapper = DenseToSparseWrapper(kgnn, model_name).to(device)
        wrapper.eval()
        wrappers[model_name] = wrapper

    results = {}

    for (gen_model, gen_class), data in generated.items():
        key = f'{gen_model}_class{gen_class}'
        results[key] = {}

        for cls_model, wrapper in wrappers.items():
            preds, probs = classify_batch(wrapper, data['adjs'], data['xs'],
                                          device)
            class0_rate = float((preds == 0).mean())
            class1_rate = float((preds == 1).mean())
            target_rate = float((preds == gen_class).mean())

            results[key][f'classified_by_{cls_model}'] = {
                'class0_rate': class0_rate,
                'class1_rate': class1_rate,
                'target_agreement': target_rate,
                'mean_target_prob': float(probs[:, gen_class].mean()),
            }

    # Compute agreement rates
    agreement = {}
    for (gen_model, gen_class), data in generated.items():
        key = f'{gen_model}_class{gen_class}'
        pred_1gnn = results[key]['classified_by_1gnn']['target_agreement']
        pred_12gnn = results[key]['classified_by_12gnn']['target_agreement']
        agreement[key] = {
            'same_model': results[key][f'classified_by_{gen_model}']['target_agreement'],
            'cross_model': results[key][f'classified_by_{"12gnn" if gen_model == "1gnn" else "1gnn"}']['target_agreement'],
        }
    results['agreement'] = agreement

    # Plot heatmap
    plot_cross_classification(results, handler, save_dir)

    return results


def plot_cross_classification(results, handler, save_dir):
    """Create cross-classification visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_names = handler.class_names
    generators = ['1gnn_class0', '1gnn_class1', '12gnn_class0', '12gnn_class1']
    classifiers = ['1gnn', '12gnn']

    # (a) Heatmap of target-class prediction rates
    ax = axes[0]
    data_matrix = np.zeros((len(generators), len(classifiers)))
    for i, gen in enumerate(generators):
        for j, cls in enumerate(classifiers):
            data_matrix[i, j] = results[gen][f'classified_by_{cls}']['target_agreement']

    im = ax.imshow(data_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels([f'Classified by\n{c.upper()}' for c in classifiers])
    ax.set_yticks(range(len(generators)))
    gen_labels = []
    for g in generators:
        model, cls = g.rsplit('_class', 1)
        gen_labels.append(f'{model.upper()}\nclass {cls} ({class_names[int(cls)]})')
    ax.set_yticklabels(gen_labels, fontsize=9)
    ax.set_title('Target-Class Prediction Rate', fontweight='bold')

    for i in range(len(generators)):
        for j in range(len(classifiers)):
            val = data_matrix[i, j]
            color = 'white' if val < 0.4 or val > 0.8 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (b) Agreement bar chart
    ax = axes[1]
    agreement = results['agreement']
    x_pos = np.arange(len(generators))
    same = [agreement[g]['same_model'] for g in generators]
    cross = [agreement[g]['cross_model'] for g in generators]

    width = 0.35
    ax.bar(x_pos - width/2, same, width, color='steelblue', label='Same Model')
    ax.bar(x_pos + width/2, cross, width, color='indianred', label='Cross Model')
    ax.set_xticks(x_pos)
    short_labels = []
    for g in generators:
        model, cls = g.rsplit('_class', 1)
        short_labels.append(f'{model.upper()}\nc{cls}')
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel('Target-Class Agreement Rate')
    ax.set_title('Same-Model vs Cross-Model Agreement', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cross_classification.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {save_path}")


# ── Analysis C: Embedding space visualization ────────────────────────


def compute_real_embeddings(kgnn, train_dataset, device, batch_size=64):
    """Compute embeddings for real graphs via sparse k-GNN."""
    loader = PyGLoader(list(train_dataset), batch_size=batch_size, shuffle=False)
    all_emb = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = kgnn.get_embedding(batch.x, batch.edge_index, batch.batch)
            all_emb.append(emb.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    return np.concatenate(all_emb), np.concatenate(all_labels)


def compute_generated_embeddings(wrapper, adjs, xs, device, batch_size=64):
    """Compute embeddings for generated graphs via dense wrapper."""
    all_emb = []
    n = adjs.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_b = torch.tensor(xs[start:end], dtype=torch.float32, device=device)
            a_b = torch.tensor(adjs[start:end], dtype=torch.float32, device=device)
            emb = wrapper.get_embedding(x_b, a_b)
            all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb)


def run_embedding_analysis(generated, kgnns, train_dataset, device, handler,
                           save_dir):
    """Compute and visualize embedding spaces for each model."""
    results = {}

    for model_name, kgnn in kgnns.items():
        print(f"    Computing embeddings for {model_name.upper()}...")

        # Real embeddings
        real_emb, real_labels = compute_real_embeddings(kgnn, train_dataset,
                                                        device)

        # Generated embeddings
        wrapper = DenseToSparseWrapper(kgnn, model_name).to(device)
        wrapper.eval()

        gen_emb_c0 = compute_generated_embeddings(
            wrapper, generated[(model_name, 0)]['adjs'],
            generated[(model_name, 0)]['xs'], device)
        gen_emb_c1 = compute_generated_embeddings(
            wrapper, generated[(model_name, 1)]['adjs'],
            generated[(model_name, 1)]['xs'], device)

        # Combine for t-SNE
        all_emb = np.vstack([real_emb, gen_emb_c0, gen_emb_c1])
        n_real = real_emb.shape[0]
        n_gen0 = gen_emb_c0.shape[0]

        if HAS_TSNE:
            print(f"      Running t-SNE ({all_emb.shape[0]} points, "
                  f"{all_emb.shape[1]}D)...")
            perp = min(30, all_emb.shape[0] // 4)
            reduced = TSNE(n_components=2, random_state=42,
                           perplexity=perp).fit_transform(all_emb)
        else:
            # PCA fallback
            print(f"      Running PCA (t-SNE unavailable)...")
            centered = all_emb - all_emb.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            reduced = centered @ Vt[:2].T

        real_2d = reduced[:n_real]
        gen0_2d = reduced[n_real:n_real + n_gen0]
        gen1_2d = reduced[n_real + n_gen0:]

        # Compute mean distances
        real_c0_2d = real_2d[real_labels.flatten() == 0]
        real_c1_2d = real_2d[real_labels.flatten() == 1]

        results[model_name] = {
            'embedding_dim': int(all_emb.shape[1]),
            'n_real': int(n_real),
            'n_gen_c0': int(n_gen0),
            'n_gen_c1': int(gen_emb_c1.shape[0]),
        }

        # Plot
        method = 't-SNE' if HAS_TSNE else 'PCA'
        fig, ax = plt.subplots(figsize=(10, 8))

        c0_name = handler.class_names.get(0, 'Class 0')
        c1_name = handler.class_names.get(1, 'Class 1')

        ax.scatter(real_c0_2d[:, 0], real_c0_2d[:, 1], s=15, alpha=0.25,
                   c='steelblue', label=f'Real {c0_name}')
        ax.scatter(real_c1_2d[:, 0], real_c1_2d[:, 1], s=15, alpha=0.25,
                   c='darkorange', label=f'Real {c1_name}')
        ax.scatter(gen0_2d[:, 0], gen0_2d[:, 1], s=60, alpha=0.6,
                   c='red', marker='*', edgecolors='darkred', linewidths=0.5,
                   label=f'Generated {c0_name}')
        ax.scatter(gen1_2d[:, 0], gen1_2d[:, 1], s=60, alpha=0.6,
                   c='limegreen', marker='D', edgecolors='darkgreen',
                   linewidths=0.5, label=f'Generated {c1_name}')

        ax.set_title(f'{model_name.upper()} Embedding Space ({method})',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.2)
        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')

        save_path = os.path.join(save_dir,
                                 f'embedding_{method.lower()}_{model_name}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved {save_path}")

    return results


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Compare GIN-Graph generated datasets against real data')
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=['mutag', 'proteins'],
                        help='Dataset to analyze')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of graphs per class per model')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--gin_checkpoint_dir', type=str, default='./gin_checkpoints')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dataset_name = args.dataset.lower()
    handler = get_handler(dataset_name)

    print("=" * 60)
    print(f"Generated vs Real Comparison: {dataset_name.upper()}")
    print(f"Device: {device}, Samples per class: {args.num_samples}")
    print("=" * 60)

    # Setup output directory
    out_dir = os.path.join(args.output_dir, dataset_name, 'comparison')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ── Step 1: Load models ──────────────────────────────────────────
    print("\n[1/5] Loading models...")
    kgnns, trainers, train_dataset, data_config, class_stats = load_all_models(
        dataset_name, args.checkpoint_dir, args.gin_checkpoint_dir, device)

    # ── Step 2: Generate datasets ────────────────────────────────────
    print(f"\n[2/5] Generating {args.num_samples} graphs per class per model...")
    t0 = time.time()
    generated = generate_all(trainers, args.num_samples)
    gen_time = time.time() - t0
    print(f"  Generation took {gen_time:.1f}s")

    # Save generated datasets
    for model_name in ['1gnn', '12gnn']:
        npz_path = os.path.join(out_dir, f'{model_name}_generated.npz')
        np.savez_compressed(
            npz_path,
            adjs_class0=generated[(model_name, 0)]['adjs'],
            xs_class0=generated[(model_name, 0)]['xs'],
            adjs_class1=generated[(model_name, 1)]['adjs'],
            xs_class1=generated[(model_name, 1)]['xs'],
        )
        print(f"  Saved {npz_path}")

    # ── Step 3: Structural comparison ────────────────────────────────
    print("\n[3/5] Structural property comparison...")
    structural_results = run_structural_analysis(
        generated, train_dataset, handler, data_config, fig_dir)

    # ── Step 4: Cross-model classification ───────────────────────────
    print("\n[4/5] Cross-model classification...")
    cross_results = run_cross_classification(
        generated, kgnns, data_config, device, handler, fig_dir)

    # ── Step 5: Embedding space visualization ────────────────────────
    print("\n[5/5] Embedding space visualization...")
    embedding_results = run_embedding_analysis(
        generated, kgnns, train_dataset, device, handler, fig_dir)

    # ── Save report ──────────────────────────────────────────────────
    report = {
        'dataset': dataset_name,
        'num_samples_per_class': args.num_samples,
        'generation_time_seconds': round(gen_time, 1),
        'structural_comparison': {
            f'{m}_class{c}': v
            for (m, c), v in structural_results.items()
        },
        'cross_classification': {
            k: v for k, v in cross_results.items()
            if k != 'agreement'
        },
        'cross_agreement': cross_results.get('agreement', {}),
        'embedding_analysis': embedding_results,
    }

    report_path = os.path.join(out_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nStructural Comparison (KS test on degree distribution):")
    for (model_name, class_idx), res in structural_results.items():
        class_label = get_class_name(class_idx, dataset_name)
        print(f"  {model_name:6s} class {class_idx} ({class_label:12s}): "
              f"KS={res['degree_ks_stat']:.3f} (p={res['degree_ks_pvalue']:.3g}), "
              f"real deg={res['real_mean_degree']:.2f}, "
              f"gen deg={res['gen_mean_degree']:.2f}")

    print("\nCross-Model Classification (target-class agreement):")
    for gen_key in ['1gnn_class0', '1gnn_class1', '12gnn_class0', '12gnn_class1']:
        model, cls = gen_key.rsplit('_class', 1)
        class_label = get_class_name(int(cls), dataset_name)
        r = cross_results[gen_key]
        same = r[f'classified_by_{model}']['target_agreement']
        other = '12gnn' if model == '1gnn' else '1gnn'
        cross = r[f'classified_by_{other}']['target_agreement']
        print(f"  {model:6s} class {cls} ({class_label:12s}): "
              f"same-model={same:.0%}, cross-model={cross:.0%}")

    print(f"\nAll outputs saved to: {out_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
