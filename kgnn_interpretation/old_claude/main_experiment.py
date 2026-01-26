"""
Main experiment script for Hierarchical k-GNN with GIN-based interpretability.

This script demonstrates:
1. Training hierarchical k-GNNs (1-2-3-GNN) on graph classification tasks
2. Analyzing how different k levels capture graph structures
3. Layer-wise interpretability using GIN embeddings
4. Testing expressiveness on graphs that require higher-order reasoning
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

from kgnn import HierarchicalKGNN, Hierarchical123GNN, OneGNN
from layer_analysis import (
    LayerEmbeddingAnalyzer,
    GraphStructureDistinguisher,
    ExpressivenessAnalyzer,
    GINGraphExplainer,
    KLevelInterpretabilityReport
)
from trainer import (
    GraphDatasetLoader,
    HierarchicalKGNNTrainer,
    create_synthetic_dataset
)

def run_expressiveness_experiment(args):
    """
    Experiment 1: Test how k affects expressiveness on challenging graphs.
    
    Creates graphs that require higher k to distinguish and tests
    models with different max_k values.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Expressiveness Analysis")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create challenging dataset
    print("\nCreating synthetic dataset (WL-hard graphs)...")
    dataset = create_synthetic_dataset('wl_hard', num_graphs=200)
    
    # Get dataset info
    num_features = dataset[0].x.size(1)
    num_classes = 2
    
    results = {}
    
    # Test different k values
    for max_k in [1, 2, 3]:
        print(f"\n--- Testing max_k = {max_k} ---")
        
        if max_k == 1:
            # Standard GNN (1-GNN only)
            model = OneGNN(
                in_channels=num_features,
                hidden_channels=args.hidden_dim,
                out_channels=num_classes,
                num_layers=args.num_layers
            )
            # Wrap to match interface
            class OneGNNWrapper(nn.Module):
                def __init__(self, gnn, hidden_dim, num_classes):
                    super().__init__()
                    self.gnn = gnn
                    self.classifier = nn.Linear(hidden_dim, num_classes)
                    
                def forward(self, x, edge_index, batch, return_layer_info=False):
                    from torch_geometric.nn import global_mean_pool
                    h = self.gnn(x, edge_index)
                    h = global_mean_pool(h, batch)
                    out = self.classifier(h)
                    if return_layer_info:
                        return out, {'1-gnn': [[self.gnn.layer_representations]]}
                    return out
            
            model = OneGNNWrapper(model, args.hidden_dim, num_classes)
        else:
            model = HierarchicalKGNN(
                num_node_features=num_features,
                hidden_channels=args.hidden_dim,
                num_classes=num_classes,
                max_k=max_k,
                num_layers_1gnn=args.num_layers,
                num_layers_kgnn=2
            )
        
        model = model.to(device)
        
        # Train
        trainer = HierarchicalKGNNTrainer(model, device)
        cv_results = trainer.cross_validate(
            dataset,
            num_folds=5,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            verbose=False
        )
        
        results[f'k={max_k}'] = {
            'mean_acc': cv_results['mean_test_acc'],
            'std_acc': cv_results['std_test_acc']
        }
        
        print(f"k={max_k}: {cv_results['mean_test_acc']:.4f} ± {cv_results['std_test_acc']:.4f}")
    
    print("\n--- Results Summary ---")
    for k, res in results.items():
        print(f"{k}: {res['mean_acc']:.4f} ± {res['std_acc']:.4f}")
    
    return results


def run_layer_analysis_experiment(args):
    """
    Experiment 2: Layer-wise analysis of what each k-level learns.
    
    Trains a 1-2-3-GNN and analyzes embeddings at each layer.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Layer-wise Analysis")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Load dataset
    loader = GraphDatasetLoader(args.data_root)
    dataset, num_features, num_classes = loader.load(args.dataset)
    
    print(f"Dataset: {args.dataset}")
    print(f"Num graphs: {len(dataset)}")
    print(f"Num features: {num_features}")
    print(f"Num classes: {num_classes}")
    
    # Create model
    model = Hierarchical123GNN(
        num_node_features=num_features,
        hidden_channels=args.hidden_dim,
        num_classes=num_classes
    ).to(device)
    
    # Train
    from torch_geometric.loader import DataLoader
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    trainer = HierarchicalKGNNTrainer(model, device)
    trainer.train(
        train_loader, test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        monitor_layers=True
    )
    
    # Generate interpretability report
    print("\nGenerating interpretability report...")
    report_generator = KLevelInterpretabilityReport(model, test_data[:50])
    report = report_generator.generate_report()
    report_generator.print_report(report)
    
    # Analyze expressiveness
    print("\nAnalyzing layer expressiveness...")
    analyzer = ExpressivenessAnalyzer(model)
    expressiveness = analyzer.measure_layer_expressiveness(test_data[:50])
    
    print("\nExpressiveness metrics by k-level:")
    for k_level, metrics in expressiveness.items():
        print(f"\n{k_level}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
    
    return {
        'report': report,
        'expressiveness': expressiveness,
        'training_history': trainer.training_history
    }


def run_structure_distinguishability_test(args):
    """
    Experiment 3: Test which graph structures can be distinguished at each k.
    
    Uses carefully constructed graph pairs known to require different k values.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Structure Distinguishability Test")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Create a pre-trained model (using synthetic data for quick testing)
    dataset = create_synthetic_dataset('motif', num_graphs=300)
    num_features = dataset[0].x.size(1)
    
    model = Hierarchical123GNN(
        num_node_features=num_features,
        hidden_channels=args.hidden_dim,
        num_classes=4
    ).to(device)
    
    # Quick training
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    trainer = HierarchicalKGNNTrainer(model, device)
    trainer.train(loader, num_epochs=50, learning_rate=args.lr, verbose=False)
    
    # Test distinguishability
    distinguisher = GraphStructureDistinguisher()
    
    print("\n1. Testing Regular Graphs (1-WL fails to distinguish):")
    g1, g2 = distinguisher.create_regular_graphs()
    results_regular = distinguisher.test_distinguishability(model, g1, g2)
    for level, can_distinguish in results_regular.items():
        status = "✓" if can_distinguish else "✗"
        print(f"   {level}: {status}")
    
    print("\n2. Testing Triangle vs Square (1-WL can distinguish):")
    g1, g2 = distinguisher.create_triangle_vs_square()
    results_basic = distinguisher.test_distinguishability(model, g1, g2)
    for level, can_distinguish in results_basic.items():
        status = "✓" if can_distinguish else "✗"
        print(f"   {level}: {status}")
    
    print("\n3. Testing CFI-like Graphs (require k≥3):")
    g1, g2 = distinguisher.create_cfi_pair()
    results_cfi = distinguisher.test_distinguishability(model, g1, g2)
    for level, can_distinguish in results_cfi.items():
        status = "✓" if can_distinguish else "✗"
        print(f"   {level}: {status}")
    
    return {
        'regular_graphs': results_regular,
        'triangle_vs_square': results_basic,
        'cfi_pair': results_cfi
    }


def run_benchmark_experiment(args):
    """
    Experiment 4: Benchmark on standard graph classification datasets.
    
    Compares 1-GNN vs hierarchical k-GNNs on real datasets.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Benchmark Evaluation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    datasets_to_test = ['MUTAG', 'PROTEINS', 'PTC_MR']
    
    all_results = {}
    
    for dataset_name in datasets_to_test:
        print(f"\n--- Dataset: {dataset_name} ---")
        
        try:
            loader = GraphDatasetLoader(args.data_root)
            dataset, num_features, num_classes = loader.load(dataset_name)
        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")
            continue
        
        stats = loader.get_dataset_stats(dataset)
        print(f"Graphs: {stats['num_graphs']}, Avg nodes: {stats['avg_nodes']:.1f}")
        
        dataset_results = {}
        
        # Test 1-GNN
        print("Testing 1-GNN...")
        model_1gnn = HierarchicalKGNN(
            num_node_features=num_features,
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            max_k=1
        ).to(device)
        
        trainer = HierarchicalKGNNTrainer(model_1gnn, device)
        cv_1gnn = trainer.cross_validate(
            dataset, num_folds=5, batch_size=args.batch_size,
            num_epochs=args.epochs, learning_rate=args.lr, verbose=False
        )
        dataset_results['1-GNN'] = {
            'acc': cv_1gnn['mean_test_acc'],
            'std': cv_1gnn['std_test_acc']
        }
        print(f"1-GNN: {cv_1gnn['mean_test_acc']:.4f} ± {cv_1gnn['std_test_acc']:.4f}")
        
        # Test 1-2-GNN
        print("Testing 1-2-GNN...")
        model_12gnn = HierarchicalKGNN(
            num_node_features=num_features,
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            max_k=2
        ).to(device)
        
        trainer = HierarchicalKGNNTrainer(model_12gnn, device)
        cv_12gnn = trainer.cross_validate(
            dataset, num_folds=5, batch_size=args.batch_size,
            num_epochs=args.epochs, learning_rate=args.lr, verbose=False
        )
        dataset_results['1-2-GNN'] = {
            'acc': cv_12gnn['mean_test_acc'],
            'std': cv_12gnn['std_test_acc']
        }
        print(f"1-2-GNN: {cv_12gnn['mean_test_acc']:.4f} ± {cv_12gnn['std_test_acc']:.4f}")
        
        # Test 1-2-3-GNN (only on small graphs due to computation)
        if stats['avg_nodes'] < 30:
            print("Testing 1-2-3-GNN...")
            model_123gnn = Hierarchical123GNN(
                num_node_features=num_features,
                hidden_channels=args.hidden_dim,
                num_classes=num_classes
            ).to(device)
            
            trainer = HierarchicalKGNNTrainer(model_123gnn, device)
            cv_123gnn = trainer.cross_validate(
                dataset, num_folds=5, batch_size=args.batch_size,
                num_epochs=args.epochs, learning_rate=args.lr, verbose=False
            )
            dataset_results['1-2-3-GNN'] = {
                'acc': cv_123gnn['mean_test_acc'],
                'std': cv_123gnn['std_test_acc']
            }
            print(f"1-2-3-GNN: {cv_123gnn['mean_test_acc']:.4f} ± {cv_123gnn['std_test_acc']:.4f}")
        
        all_results[dataset_name] = dataset_results
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<15} {'1-GNN':<20} {'1-2-GNN':<20} {'1-2-3-GNN':<20}")
    print("-"*75)
    
    for dataset_name, results in all_results.items():
        row = f"{dataset_name:<15}"
        for model_name in ['1-GNN', '1-2-GNN', '1-2-3-GNN']:
            if model_name in results:
                row += f"{results[model_name]['acc']:.3f}±{results[model_name]['std']:.3f}  ".ljust(20)
            else:
                row += "N/A".ljust(20)
        print(row)
    
    return all_results


def run_gin_explanation_experiment(args):
    """
    Experiment 5: Generate model-level explanations using GIN-Graph approach.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: GIN-Graph Model-Level Explanations")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Create and train model on motif dataset
    dataset = create_synthetic_dataset('motif', num_graphs=300)
    num_features = dataset[0].x.size(1)
    
    model = Hierarchical123GNN(
        num_node_features=num_features,
        hidden_channels=args.hidden_dim,
        num_classes=4
    ).to(device)
    
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    trainer = HierarchicalKGNNTrainer(model, device)
    trainer.train(loader, num_epochs=args.epochs, learning_rate=args.lr)
    
    # Generate explanations for each class
    explainer = GINGraphExplainer(
        model, hidden_dim=32, num_nodes=8, node_features=num_features
    )
    
    print("\nGenerating explanations for each class...")
    explanations = {}
    
    for target_class in range(4):
        print(f"\nClass {target_class}:")
        best_adj, best_features = explainer.generate_explanation(
            target_class=target_class,
            num_iterations=200,
            learning_rate=0.01
        )
        
        if best_adj is not None:
            # Compute statistics
            adj_binary = (best_adj > 0.5).float().squeeze()
            num_edges = adj_binary.sum().item() / 2
            
            score, details = explainer.compute_validation_score(
                best_adj, best_features, target_class,
                reference_mean_degree=2.0, reference_std_degree=1.0
            )
            
            explanations[target_class] = {
                'num_edges': num_edges,
                'validation_score': score,
                'details': details
            }
            
            print(f"  Validation score: {score:.4f}")
            print(f"  Probability: {details['probability']:.4f}")
            print(f"  Avg degree: {details['avg_degree']:.2f}")
    
    return explanations


def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical k-GNN Experiments with GIN-based Interpretability'
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='Dataset name for experiments')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in 1-GNN')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    # Experiment selection
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'expressiveness', 'layer_analysis', 
                                'distinguishability', 'benchmark', 'explanation'],
                        help='Which experiment to run')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run experiments
    results = {}
    
    if args.experiment in ['all', 'expressiveness']:
        results['expressiveness'] = run_expressiveness_experiment(args)
    
    if args.experiment in ['all', 'layer_analysis']:
        results['layer_analysis'] = run_layer_analysis_experiment(args)
    
    if args.experiment in ['all', 'distinguishability']:
        results['distinguishability'] = run_structure_distinguishability_test(args)
    
    if args.experiment in ['all', 'benchmark']:
        results['benchmark'] = run_benchmark_experiment(args)
    
    if args.experiment in ['all', 'explanation']:
        results['explanation'] = run_gin_explanation_experiment(args)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'results_{args.experiment}_{timestamp}.json'
    
    # Convert non-serializable objects
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    return results


if __name__ == '__main__':
    main()
