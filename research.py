"""
research.py
Checkpoint-based analysis script — no training, only evaluation and visualization.

Two subcommands:
  gnn  — Load a k-GNN checkpoint, evaluate on the test set, save report.
  gin  — Load k-GNN + GIN-Graph checkpoints, generate explanations, save figures + report.

Usage:
    python research.py gnn --model 1gnn --dataset mutag
    python research.py gin --model 1gnn --dataset mutag --target_class 0
    python research.py gin --model 123gnn --dataset mutag --target_class 0 --num_samples 200
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')

from data_loader import (
    load_dataset, create_data_loaders, get_class_statistics,
    get_class_subset, AVAILABLE_DATASETS
)
from models_kgnn import get_model, count_parameters
from config import DataConfig, GINGraphConfig, get_class_name
from train_kgnn import evaluate
from train_gin_graph import load_pretrained_kgnn, GINGraphTrainer
from visualize import (
    plot_explanation_grid, plot_training_history,
    plot_validation_scores_distribution, plot_legend
)


# ── helpers ──────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def _load_kgnn_checkpoint(checkpoint_dir, dataset_name, model_name, device):
    """Load a k-GNN checkpoint and return (model, checkpoint_dict)."""
    path = os.path.join(checkpoint_dir, f'{dataset_name}_{model_name}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Train first: python train_kgnn.py --dataset {dataset_name} --model {model_name}"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = get_model(
        ckpt['model_name'],
        input_dim=ckpt['input_dim'],
        hidden_dim=ckpt['hidden_dim'],
        output_dim=ckpt['output_dim']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, ckpt


# ── gnn subcommand ───────────────────────────────────────────

def cmd_gnn(args):
    device = _resolve_device(args.device)
    dataset_name = args.dataset.lower()

    print("=" * 60)
    print(f"research.py gnn — {args.model.upper()} on {dataset_name.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    # Load checkpoint
    model, ckpt = _load_kgnn_checkpoint(
        args.checkpoint_dir, dataset_name, args.model, device
    )
    num_params = ckpt.get('num_params', count_parameters(model))
    print(f"Loaded {args.model.upper()} — "
          f"Best test acc: {ckpt['best_test_acc']:.4f} (epoch {ckpt['best_epoch']}), "
          f"Params: {num_params:,}")

    # Load dataset + create loaders (same split as training)
    data_config = DataConfig.from_dataset(dataset_name)
    dataset = load_dataset(dataset_name, data_config.root)
    train_loader, test_loader, train_ds, test_ds = create_data_loaders(
        dataset,
        train_ratio=data_config.train_ratio,
        batch_size=data_config.batch_size,
        seed=data_config.seed
    )
    print(f"Dataset: {len(dataset)} graphs (train {len(train_ds)}, test {len(test_ds)})")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)

    print()
    print("-" * 40)
    print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
    print(f"  Test  Loss: {test_loss:.4f}  |  Test  Acc: {test_acc:.4f}")
    print(f"  Best  Acc (from ckpt):  {ckpt['best_test_acc']:.4f} (epoch {ckpt['best_epoch']})")
    print("-" * 40)

    # Save report
    out_dir = os.path.join(args.output_dir, dataset_name, args.model)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        'model': args.model,
        'dataset': dataset_name,
        'num_params': num_params,
        'hidden_dim': ckpt['hidden_dim'],
        'input_dim': ckpt['input_dim'],
        'output_dim': ckpt['output_dim'],
        'train_loss': float(train_loss),
        'train_acc': float(train_acc),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'best_test_acc': float(ckpt['best_test_acc']),
        'best_epoch': int(ckpt['best_epoch']),
    }

    report_path = os.path.join(out_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")


# ── gin subcommand ───────────────────────────────────────────

def cmd_gin(args):
    device = _resolve_device(args.device)
    dataset_name = args.dataset.lower()
    target_class = args.target_class

    class_label = get_class_name(target_class, dataset_name)

    print("=" * 60)
    print(f"research.py gin — {args.model.upper()} on {dataset_name.upper()}, "
          f"class {target_class} ({class_label})")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    # 1. Load k-GNN
    print("Loading k-GNN checkpoint...")
    pretrained_gnn = load_pretrained_kgnn(
        args.model, args.checkpoint_dir, device, dataset_name
    )

    # 2. Load dataset + class stats (training data only to avoid leakage)
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    data_config = DataConfig.from_dataset(dataset_name)
    _, _, train_dataset, _ = create_data_loaders(
        dataset, seed=data_config.seed
    )
    class_stats = get_class_statistics(train_dataset)
    for label in [0, 1]:
        nodes = [d.num_nodes for d in train_dataset if d.y.item() == label]
        class_stats[label]['avg_nodes'] = np.mean(nodes) if nodes else 0

    # 3. Locate GIN checkpoint
    gin_path = os.path.join(
        args.gin_checkpoint_dir, dataset_name,
        f'{args.model}_class{target_class}.pt'
    )
    if not os.path.exists(gin_path):
        raise FileNotFoundError(
            f"GIN checkpoint not found: {gin_path}\n"
            f"Train first: python train_gin_graph.py --dataset {dataset_name} "
            f"--model {args.model} --target_class {target_class}"
        )

    # 4. Reconstruct trainer and load checkpoint
    # Read config from checkpoint to match hidden_dim used during training
    gin_ckpt = torch.load(gin_path, map_location=device, weights_only=False)
    gin_config = gin_ckpt.get('config', GINGraphConfig())

    trainer = GINGraphTrainer(
        pretrained_gnn=pretrained_gnn,
        model_type=args.model,
        target_class=target_class,
        config=gin_config,
        data_config=data_config,
        device=device,
        class_stats=class_stats,
        dataset_name=dataset_name
    )
    trainer.load_checkpoint(gin_path)
    print(f"Loaded GIN checkpoint: {gin_path}")
    print(f"  Trained for {trainer.epoch} epochs, {trainer.global_step} steps")

    # Compute class centroid if not in checkpoint (backwards compat)
    if trainer.class_centroid is None:
        print("  Computing class centroid from training data...")
        target_dataset = get_class_subset(train_dataset, target_class)
        trainer.compute_class_centroid(target_dataset)
    print()

    # 5. Generate explanations
    num_samples = args.num_samples
    print(f"Generating {num_samples} explanations...")
    adjs, xs, metrics = trainer.generate_explanations(num_samples=num_samples)

    # 6. Compute summary + best explanations
    summary = trainer.evaluator.compute_summary_stats(metrics)
    best = trainer.evaluator.get_best_explanations(metrics, top_k=10)

    print(f"\nGeneration Summary:")
    print(f"  Total generated: {summary['total_generated']}")
    print(f"  Valid explanations: {summary['num_valid']} "
          f"({summary['validity_rate']*100:.1f}%)")
    print(f"  Mean validation score: {summary['mean_validation_score']:.4f}")
    print(f"  Mean prediction prob:  {summary['mean_prediction_prob']:.4f}")

    if best:
        print(f"\nTop {min(5, len(best))} Explanations:")
        for rank, (idx, m) in enumerate(best[:5], 1):
            print(f"  {rank}. Score: {m.validation_score:.4f}, "
                  f"Nodes: {m.num_nodes}, Edges: {m.num_edges}, "
                  f"Pred: {m.prediction_probability:.4f}")

    # 7. Save outputs
    out_dir = os.path.join(
        args.output_dir, dataset_name,
        f'{args.model}_class{target_class}'
    )
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Explanations grid (top-k)
    if best:
        best_indices = [idx for idx, _ in best]
        best_adjs = adjs[best_indices]
        best_xs = xs[best_indices]
        best_metrics = [m for _, m in best]
        plot_explanation_grid(
            best_adjs, best_xs, metrics=best_metrics,
            save_path=os.path.join(fig_dir, 'explanations.png'),
            title=f'{args.model.upper()} — Class {target_class} ({class_label}) '
                  f'Best Explanations',
            dataset=dataset_name
        )

    # Training curves
    if trainer.history and trainer.history.get('d_loss'):
        plot_training_history(
            trainer.history,
            save_path=os.path.join(fig_dir, 'training.png')
        )

    # Validation score distributions
    plot_validation_scores_distribution(
        metrics,
        save_path=os.path.join(fig_dir, 'metrics.png')
    )

    # Node-type legend
    plot_legend(
        dataset_name,
        save_path=os.path.join(fig_dir, f'{dataset_name}_legend.png')
    )

    # Save best explanations as npz
    if best:
        np.savez(
            os.path.join(out_dir, 'explanations.npz'),
            adjs=adjs[best_indices],
            xs=xs[best_indices],
            indices=best_indices
        )

    # Save report
    report = {
        'model': args.model,
        'dataset': dataset_name,
        'target_class': target_class,
        'class_label': class_label,
        'num_samples': num_samples,
        'trainer_epoch': int(trainer.epoch),
        'trainer_steps': int(trainer.global_step),
        'total_generated': summary['total_generated'],
        'num_valid': summary['num_valid'],
        'validity_rate': float(summary['validity_rate']),
        'mean_validation_score': float(summary['mean_validation_score']),
        'mean_valid_score': float(summary['mean_valid_score']),
        'mean_prediction_prob': float(summary['mean_prediction_prob']),
        'mean_embedding_sim': float(summary['mean_embedding_sim']),
        'mean_degree_score': float(summary['mean_degree_score']),
        'mean_granularity': float(summary['mean_granularity']),
        'top_explanations': [
            {
                'rank': rank,
                'index': int(idx),
                'validation_score': float(m.validation_score),
                'prediction_probability': float(m.prediction_probability),
                'num_nodes': m.num_nodes,
                'num_edges': m.num_edges,
                'degree_score': float(m.degree_score),
                'granularity': float(m.granularity),
            }
            for rank, (idx, m) in enumerate(best, 1)
        ],
    }

    report_path = os.path.join(out_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {out_dir}/")
    print(f"  figures/explanations.png")
    if trainer.history and trainer.history.get('d_loss'):
        print(f"  figures/training.png")
    print(f"  figures/metrics.png")
    print(f"  figures/{dataset_name}_legend.png")
    print(f"  explanations.npz")
    print(f"  report.json")


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Checkpoint-based analysis — no training, only evaluation and visualization'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ── shared arguments ──
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--dataset', type=str, default='mutag',
                        choices=AVAILABLE_DATASETS,
                        help=f'Dataset ({", ".join(AVAILABLE_DATASETS)})')
    shared.add_argument('--model', type=str, default='1gnn',
                        choices=['1gnn', '12gnn', '123gnn'],
                        help='k-GNN model name')
    shared.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    shared.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory with k-GNN checkpoints')
    shared.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for reports and figures')

    # ── gnn subcommand ──
    sub_gnn = subparsers.add_parser(
        'gnn', parents=[shared],
        help='Evaluate a k-GNN checkpoint on the test set'
    )
    sub_gnn.set_defaults(func=cmd_gnn)

    # ── gin subcommand ──
    sub_gin = subparsers.add_parser(
        'gin', parents=[shared],
        help='Generate explanations from a GIN-Graph checkpoint'
    )
    sub_gin.add_argument('--target_class', type=int, default=0,
                         help='Target class for explanation generation')
    sub_gin.add_argument('--num_samples', type=int, default=100,
                         help='Number of explanation samples to generate')
    sub_gin.add_argument('--gin_checkpoint_dir', type=str,
                         default='./gin_checkpoints',
                         help='Directory with GIN-Graph checkpoints')
    sub_gin.set_defaults(func=cmd_gin)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
