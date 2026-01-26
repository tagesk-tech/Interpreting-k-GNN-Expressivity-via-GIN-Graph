"""
run_experiment.py
Main experiment runner for k-GNN interpretation study.

This script orchestrates the full pipeline:
1. Train k-GNN models (1-GNN, 1-2-GNN, 1-2-3-GNN)
2. Train GIN-Graph generators for each model
3. Generate and evaluate explanations
4. Compare interpretation quality across k values

Usage:
    python run_experiment.py                    # Run full experiment
    python run_experiment.py --skip_training    # Skip k-GNN training (use existing checkpoints)
    python run_experiment.py --models 1gnn 12gnn  # Only specific models
"""

import torch
import numpy as np
import argparse
import os
import json
import time
from datetime import datetime

from data_loader import load_mutag, create_data_loaders, get_class_statistics, get_dataset_statistics
from models_kgnn import get_model, count_parameters
from train_kgnn import train_single_model
from train_gin_graph import GINGraphTrainer, load_pretrained_kgnn
from visualize import (
    plot_explanation_grid, 
    plot_training_history,
    plot_validation_scores_distribution,
    create_comparison_figure,
    plot_atom_legend
)
from metrics import ExplanationEvaluator
from config import ExperimentConfig, CLASS_NAMES


def setup_experiment(args):
    """Setup experiment directories and configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = ExperimentConfig(
        checkpoint_dir=args.checkpoint_dir,
        results_dir=os.path.join(args.output_dir, f'experiment_{timestamp}'),
        figures_dir=os.path.join(args.output_dir, f'experiment_{timestamp}', 'figures'),
        device=args.device
    )
    
    # Override from args
    config.kgnn.epochs = args.kgnn_epochs
    config.kgnn.hidden_dim = args.hidden_dim
    config.gin_graph.epochs = args.gin_epochs
    config.gin_graph.hidden_dim = args.hidden_dim
    config.target_class = args.target_class
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)
    
    return config


def run_kgnn_training(config, models_to_train, dataset, train_loader, test_loader, device):
    """Train all k-GNN models."""
    print("\n" + "=" * 70)
    print("PHASE 1: Training k-GNN Models")
    print("=" * 70)
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print('='*60)
        
        model, acc = train_single_model(
            model_name, config, device,
            dataset, train_loader, test_loader,
            verbose=True
        )
        
        results[model_name] = {
            'accuracy': acc,
            'num_params': count_parameters(model)
        }
    
    return results


def run_gin_graph_training(config, models_to_train, dataset, device, class_stats):
    """Train GIN-Graph generators for each k-GNN model."""
    print("\n" + "=" * 70)
    print("PHASE 2: Training GIN-Graph Generators")
    print("=" * 70)
    
    target_class = config.target_class
    target_dataset = [d for d in dataset if d.y.item() == target_class]
    
    print(f"\nTarget class: {target_class} ({CLASS_NAMES[target_class]})")
    print(f"Training samples: {len(target_dataset)}")
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"GIN-Graph for {model_name.upper()}")
        print('='*60)
        
        # Load pretrained k-GNN
        pretrained_gnn = load_pretrained_kgnn(model_name, config.checkpoint_dir, device)
        
        # Create trainer
        trainer = GINGraphTrainer(
            pretrained_gnn=pretrained_gnn,
            model_type=model_name,
            target_class=target_class,
            config=config.gin_graph,
            data_config=config.data,
            device=device,
            class_stats=class_stats
        )
        
        # Train
        trainer.train(target_dataset, epochs=config.gin_graph.epochs, log_interval=50)
        
        # Generate explanations
        print("\nGenerating explanations...")
        adjs, xs, metrics = trainer.generate_explanations(num_samples=100)
        
        # Compute summary
        summary = trainer.evaluator.compute_summary_stats(metrics)
        
        # Get best explanations
        best = trainer.evaluator.get_best_explanations(metrics, top_k=10)
        best_indices = [idx for idx, _ in best]
        best_metrics = [m for _, m in best]
        
        results[model_name] = {
            'adjs': adjs,
            'xs': xs,
            'metrics': metrics,
            'summary': summary,
            'best_indices': best_indices,
            'best_metrics': best_metrics,
            'history': trainer.history
        }
        
        # Save checkpoint
        save_path = os.path.join(config.results_dir, f'gin_graph_{model_name}.pt')
        trainer.save_checkpoint(save_path)
        
        print(f"\nResults for {model_name.upper()}:")
        print(f"  Valid explanations: {summary['num_valid']}/{summary['total_generated']}")
        print(f"  Mean validation score: {summary['mean_validation_score']:.4f}")
        print(f"  Mean prediction prob: {summary['mean_prediction_prob']:.4f}")
    
    return results


def generate_figures(config, gin_results, models_to_train):
    """Generate all figures for the experiment."""
    print("\n" + "=" * 70)
    print("PHASE 3: Generating Figures")
    print("=" * 70)
    
    # Atom type legend
    plot_atom_legend(os.path.join(config.figures_dir, 'atom_legend.png'))
    
    for model_name in models_to_train:
        data = gin_results[model_name]
        
        # Best explanations grid
        best_idx = data['best_indices'][:10]
        plot_explanation_grid(
            data['adjs'][best_idx],
            data['xs'][best_idx],
            [data['metrics'][i] for i in best_idx],
            num_cols=5,
            save_path=os.path.join(config.figures_dir, f'explanations_{model_name}.png'),
            title=f'Best Explanations for {model_name.upper()}'
        )
        
        # Training history
        plot_training_history(
            data['history'],
            save_path=os.path.join(config.figures_dir, f'training_{model_name}.png')
        )
        
        # Metrics distribution
        plot_validation_scores_distribution(
            data['metrics'],
            save_path=os.path.join(config.figures_dir, f'metrics_{model_name}.png')
        )
    
    # Comparison across models
    if len(models_to_train) > 1:
        comparison_data = {
            name: {
                'adjs': gin_results[name]['adjs'],
                'xs': gin_results[name]['xs'],
                'metrics': gin_results[name]['metrics']
            }
            for name in models_to_train
        }
        create_comparison_figure(
            comparison_data,
            save_path=os.path.join(config.figures_dir, 'model_comparison.png')
        )
    
    print(f"\nFigures saved to: {config.figures_dir}")


def generate_report(config, kgnn_results, gin_results, models_to_train):
    """Generate final experiment report."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'target_class': config.target_class,
        'target_class_name': CLASS_NAMES[config.target_class],
        'models': {}
    }
    
    print(f"\nTarget Class: {config.target_class} ({CLASS_NAMES[config.target_class]})")
    print("\n" + "-" * 70)
    print(f"{'Model':<10} | {'Params':<10} | {'Test Acc':<10} | {'Valid %':<10} | {'Val Score':<10} | {'Pred Prob':<10}")
    print("-" * 70)
    
    for model_name in models_to_train:
        kgnn = kgnn_results.get(model_name, {})
        gin = gin_results.get(model_name, {})
        summary = gin.get('summary', {})
        
        acc = kgnn.get('accuracy', 0)
        params = kgnn.get('num_params', 0)
        valid_pct = summary.get('validity_rate', 0) * 100
        val_score = summary.get('mean_validation_score', 0)
        pred_prob = summary.get('mean_prediction_prob', 0)
        
        print(f"{model_name.upper():<10} | {params:<10,} | {acc:<10.4f} | {valid_pct:<10.1f} | {val_score:<10.4f} | {pred_prob:<10.4f}")
        
        report['models'][model_name] = {
            'kgnn': kgnn,
            'gin_graph_summary': summary,
            'num_valid': summary.get('num_valid', 0),
            'mean_validation_score': val_score,
            'mean_prediction_prob': pred_prob,
        }
    
    print("-" * 70)
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if len(models_to_train) > 1:
        # Compare models
        scores = {name: gin_results[name]['summary']['mean_validation_score'] for name in models_to_train}
        best_model = max(scores, key=scores.get)
        
        print(f"\nHigher-order GNN interpretation analysis:")
        print(f"  Best explanation quality: {best_model.upper()} (score: {scores[best_model]:.4f})")
        
        if '123gnn' in models_to_train and '1gnn' in models_to_train:
            improvement = (scores['123gnn'] - scores['1gnn']) / scores['1gnn'] * 100
            print(f"  1-2-3-GNN vs 1-GNN improvement: {improvement:+.1f}%")
        
        # Validity rates
        valid_rates = {name: gin_results[name]['summary']['validity_rate'] for name in models_to_train}
        print(f"\nValidity rates:")
        for name, rate in valid_rates.items():
            print(f"  {name.upper()}: {rate*100:.1f}%")
    
    # Save report
    report_path = os.path.join(config.results_dir, 'experiment_report.json')
    with open(report_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        clean_report = json.loads(json.dumps(report, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        json.dump(clean_report, f, indent=2)
    
    print(f"\nFull report saved to: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Run k-GNN interpretation experiment')
    
    # Model selection
    parser.add_argument('--models', nargs='+', default=['1gnn', '12gnn', '123gnn'],
                        choices=['1gnn', '12gnn', '123gnn'],
                        help='Models to train and interpret')
    parser.add_argument('--target_class', type=int, default=0,
                        choices=[0, 1],
                        help='Target class for interpretation')
    
    # Training settings
    parser.add_argument('--kgnn_epochs', type=int, default=100,
                        help='Epochs for k-GNN training')
    parser.add_argument('--gin_epochs', type=int, default=300,
                        help='Epochs for GIN-Graph training')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    
    # Workflow control
    parser.add_argument('--skip_kgnn_training', action='store_true',
                        help='Skip k-GNN training (use existing checkpoints)')
    parser.add_argument('--skip_gin_training', action='store_true',
                        help='Skip GIN-Graph training (use existing results)')
    
    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for k-GNN checkpoints')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    config = setup_experiment(args)
    device = config.get_device()
    
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 70)
    print("k-GNN Interpretation Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Target class: {args.target_class} ({CLASS_NAMES[args.target_class]})")
    print(f"Output: {config.results_dir}")
    print()
    
    # Load dataset
    print("Loading MUTAG dataset...")
    dataset = load_mutag(config.data.root)
    stats = get_dataset_statistics(dataset)
    
    train_loader, test_loader, _, _ = create_data_loaders(
        dataset,
        train_ratio=config.data.train_ratio,
        batch_size=config.data.batch_size,
        seed=config.data.seed
    )
    
    # Get class statistics
    class_stats = get_class_statistics(dataset)
    for label in [0, 1]:
        nodes = [d.num_nodes for d in dataset if d.y.item() == label]
        class_stats[label]['avg_nodes'] = np.mean(nodes)
    
    print(f"Dataset: {stats['num_graphs']} graphs, {stats['num_classes']} classes")
    print()
    
    # Phase 1: k-GNN Training
    if args.skip_kgnn_training:
        print("Skipping k-GNN training (using existing checkpoints)")
        kgnn_results = {}
        for model_name in args.models:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'{model_name}.pt')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                kgnn_results[model_name] = {
                    'accuracy': checkpoint.get('best_test_acc', 0),
                    'num_params': checkpoint.get('num_params', 0)
                }
            else:
                print(f"Warning: Checkpoint not found for {model_name}")
    else:
        kgnn_results = run_kgnn_training(
            config, args.models, dataset, train_loader, test_loader, device
        )
    
    # Phase 2: GIN-Graph Training
    if args.skip_gin_training:
        print("\nSkipping GIN-Graph training")
        gin_results = {}
    else:
        gin_results = run_gin_graph_training(
            config, args.models, dataset, device, class_stats
        )
    
    # Phase 3: Generate figures
    if gin_results:
        generate_figures(config, gin_results, args.models)
    
    # Phase 4: Generate report
    if gin_results:
        report = generate_report(config, kgnn_results, gin_results, args.models)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {config.results_dir}")


if __name__ == "__main__":
    main()
