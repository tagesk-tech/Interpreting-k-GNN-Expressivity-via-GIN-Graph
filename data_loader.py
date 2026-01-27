"""
data_loader.py
Data loading utilities for graph classification datasets.

Supported datasets:
- MUTAG: 188 mutagenic compounds, binary classification
- DD: Large protein graphs, binary classification
- PROTEINS: Medium protein graphs, binary classification
- QM9: 130k molecules with regression targets (not suitable for classification)
"""

import torch
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import numpy as np
from collections import Counter
import ssl
from typing import Tuple, Dict, Any, Optional, Union

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context


# Available datasets for classification
AVAILABLE_DATASETS = ['mutag', 'dd', 'proteins']

# Dataset metadata for configuration
DATASET_INFO = {
    'mutag': {
        'name': 'MUTAG',
        'max_nodes': 28,
        'task': 'classification',
        'description': '188 mutagenic compounds, binary classification'
    },
    'dd': {
        'name': 'DD',
        'max_nodes': 500,  # DD has larger graphs
        'task': 'classification',
        'description': 'Large protein graphs, binary classification'
    },
    'proteins': {
        'name': 'PROTEINS',
        'max_nodes': 620,  # PROTEINS can have large graphs
        'task': 'classification',
        'description': 'Medium protein graphs, binary classification'
    },
    'qm9': {
        'name': 'QM9',
        'max_nodes': 29,
        'task': 'regression',
        'description': '130k molecules, regression (not for classification)'
    }
}


def load_dataset(name: str, root: str = './data') -> TUDataset:
    """
    Load a dataset by name.

    Args:
        name: Dataset name (mutag, dd, proteins, qm9)
        root: Directory to store/load the dataset

    Returns:
        The loaded dataset

    Raises:
        ValueError: If dataset name is not recognized
    """
    name = name.lower()

    if name == 'mutag':
        return load_mutag(root)
    elif name == 'dd':
        return load_dd(root)
    elif name == 'proteins':
        return load_proteins(root)
    elif name == 'qm9':
        return load_qm9(root)
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {', '.join(AVAILABLE_DATASETS + ['qm9'])}"
        )


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Get metadata about a dataset."""
    name = name.lower()
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_INFO[name]


def load_dd(root: str = './data') -> TUDataset:
    """Load the D&D dataset (large protein graphs)."""
    return TUDataset(root=root, name='DD')


def load_proteins(root: str = './data') -> TUDataset:
    """Load the PROTEINS dataset (medium protein graphs)."""
    return TUDataset(root=root, name='PROTEINS')


def load_qm9(root: str = './data') -> QM9:
    """
    Load the QM9 dataset (130k molecules with regression targets).
    Note: QM9 uses regression targets, not suitable for classification tasks.
    """
    return QM9(root=root)


def load_mutag(root: str = './data') -> TUDataset:
    """
    Load the MUTAG dataset.
    
    MUTAG contains 188 mutagenic aromatic and heteroaromatic nitro compounds.
    - Task: Binary classification (mutagenic vs non-mutagenic)
    - Node features: 7 discrete labels (atom types: C, N, O, F, I, Cl, Br)
    - Edge features: 4 discrete labels (bond types)
    
    Args:
        root: Directory to store/load the dataset
        
    Returns:
        dataset: The full MUTAG dataset
    """
    dataset = TUDataset(root=root, name='MUTAG')
  
    return dataset



def get_dataset_statistics(dataset: TUDataset) -> Dict[str, Any]:
    """
    Compute statistics about the dataset.
    
    Args:
        dataset: A PyTorch Geometric dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    num_nodes = [data.num_nodes for data in dataset]
    num_edges = [data.num_edges for data in dataset]
    labels = [data.y.item() for data in dataset]
    label_counts = Counter(labels)
    avg_degrees = [data.num_edges / data.num_nodes for data in dataset]
    
    stats = {
        'name': dataset.name,
        'num_graphs': len(dataset),
        'num_classes': dataset.num_classes,
        'num_node_features': dataset.num_node_features,
        'num_edge_features': dataset.num_edge_features,
        'nodes': {
            'min': min(num_nodes),
            'max': max(num_nodes),
            'mean': np.mean(num_nodes),
            'std': np.std(num_nodes)
        },
        'edges': {
            'min': min(num_edges),
            'max': max(num_edges),
            'mean': np.mean(num_edges),
            'std': np.std(num_edges)
        },
        'class_distribution': dict(label_counts),
        'avg_degree': {
            'mean': np.mean(avg_degrees),
            'std': np.std(avg_degrees)
        }
    }
    return stats


def print_dataset_statistics(stats: Dict[str, Any]) -> None:
    """Pretty print dataset statistics."""
    print("=" * 60)
    print(f"Dataset: {stats['name']}")
    print("=" * 60)
    
    print(f"\nBasic Statistics:")
    print(f"  Number of graphs: {stats['num_graphs']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Number of node features: {stats['num_node_features']}")
    print(f"  Number of edge features: {stats['num_edge_features']}")
    
    print(f"\nGraph Size Statistics:")
    print(f"  Nodes - min: {stats['nodes']['min']}, max: {stats['nodes']['max']}, "
          f"mean: {stats['nodes']['mean']:.2f}, std: {stats['nodes']['std']:.2f}")
    print(f"  Edges - min: {stats['edges']['min']}, max: {stats['edges']['max']}, "
          f"mean: {stats['edges']['mean']:.2f}, std: {stats['edges']['std']:.2f}")
    
    print(f"\nClass Distribution:")
    for label, count in sorted(stats['class_distribution'].items()):
        pct = 100 * count / stats['num_graphs']
        print(f"  Class {label}: {count} graphs ({pct:.1f}%)")
    
    print(f"\nAverage Degree: {stats['avg_degree']['mean']:.2f} "
          f"(std: {stats['avg_degree']['std']:.2f})")


def create_data_loaders(
    dataset: TUDataset,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, TUDataset, TUDataset]:
    """
    Split dataset and create train/test DataLoaders.
    
    Args:
        dataset: The full dataset
        train_ratio: Fraction of data for training
        batch_size: Batch size for DataLoader
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    torch.manual_seed(seed)
    
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * train_ratio)
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset


def get_class_subset(dataset: TUDataset, target_class: int) -> TUDataset:
    """
    Get a subset of the dataset containing only graphs of a specific class.
    
    Args:
        dataset: The full dataset
        target_class: The class label to filter for
        
    Returns:
        Filtered dataset
    """
    indices = [i for i, data in enumerate(dataset) if data.y.item() == target_class]
    return dataset[indices]


def get_class_statistics(dataset: TUDataset) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class statistics for the validation score calculation.
    
    Args:
        dataset: The dataset
        
    Returns:
        Dictionary mapping class -> {mean_degree, std_degree}
    """
    class_stats = {}
    
    for label in [0, 1]:
        degrees = []
        for data in dataset:
            if data.y.item() == label:
                avg_degree = data.num_edges / data.num_nodes
                degrees.append(avg_degree)
        
        if degrees:
            class_stats[label] = {
                'mean_degree': np.mean(degrees),
                'std_degree': np.std(degrees)
            }
    
    return class_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test data loader')
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=AVAILABLE_DATASETS,
                        help=f'Dataset to load ({", ".join(AVAILABLE_DATASETS)})')
    args = parser.parse_args()

    # Test the data loader
    print(f"Loading {args.dataset.upper()} dataset...\n")
    dataset = load_dataset(args.dataset)

    stats = get_dataset_statistics(dataset)
    print_dataset_statistics(stats)

    print("\n")
    train_loader, test_loader, train_data, test_data = create_data_loaders(
        dataset, train_ratio=0.8, batch_size=32
    )
    print(f"Train: {len(train_data)} graphs, Test: {len(test_data)} graphs")

    class_stats = get_class_statistics(dataset)
    print(f"\nPer-class degree statistics:")
    for label, stats_item in class_stats.items():
        print(f"  Class {label}: mean={stats_item['mean_degree']:.2f}, std={stats_item['std_degree']:.2f}")
