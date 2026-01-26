"""
Load and explore the MUTAG dataset for GNN experiments.
"""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import numpy as np
from collections import Counter
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_mutag(root='./data'):
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


def dataset_statistics(dataset):
    """
    Compute and print statistics about the dataset.
    
    Args:
        dataset: A PyTorch Geometric dataset
    """
    print("=" * 60)
    print(f"Dataset: {dataset.name}")
    print("=" * 60)
    
    # Basic info
    print(f"\nBasic Statistics:")
    print(f"  Number of graphs: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes}")
    print(f"  Number of node features: {dataset.num_node_features}")
    print(f"  Number of edge features: {dataset.num_edge_features}")
    
    # Compute graph-level statistics
    num_nodes = [data.num_nodes for data in dataset]
    num_edges = [data.num_edges for data in dataset]
    
    print(f"\nGraph Size Statistics:")
    print(f"  Nodes - min: {min(num_nodes)}, max: {max(num_nodes)}, "
          f"mean: {np.mean(num_nodes):.2f}, std: {np.std(num_nodes):.2f}")
    print(f"  Edges - min: {min(num_edges)}, max: {max(num_edges)}, "
          f"mean: {np.mean(num_edges):.2f}, std: {np.std(num_edges):.2f}")
    
    # Class distribution
    labels = [data.y.item() for data in dataset]
    label_counts = Counter(labels)
    print(f"\nClass Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} graphs ({100*count/len(dataset):.1f}%)")
    
    # Average degree
    avg_degrees = [data.num_edges / data.num_nodes for data in dataset]
    print(f"\nAverage Degree: {np.mean(avg_degrees):.2f} (std: {np.std(avg_degrees):.2f})")
    
    return {
        'num_graphs': len(dataset),
        'num_classes': dataset.num_classes,
        'num_node_features': dataset.num_node_features,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'labels': labels
    }


def inspect_single_graph(data, idx=0):
    """
    Inspect a single graph from the dataset.
    
    Args:
        data: A single PyTorch Geometric Data object
        idx: Index of the graph (for display purposes)
    """
    print("=" * 60)
    print(f"Graph {idx} Inspection")
    print("=" * 60)
    
    print(f"\nData object: {data}")
    print(f"\nAttributes:")
    print(f"  x (node features): shape {data.x.shape}")
    print(f"  edge_index: shape {data.edge_index.shape}")
    print(f"  edge_attr (edge features): shape {data.edge_attr.shape}")
    print(f"  y (label): {data.y.item()}")
    
    print(f"\nNode feature matrix (first 5 nodes):")
    print(data.x[:5])
    
    print(f"\nEdge index (first 10 edges):")
    print(data.edge_index[:, :10])
    
    # Convert to dense adjacency for visualization
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    print(f"\nAdjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix (top-left 5x5):")
    print(adj[:5, :5].int())


def create_data_loaders(dataset, train_ratio=0.8, batch_size=32, seed=42):
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
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Shuffle and split
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * train_ratio)
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f"\nData Split:")
    print(f"  Training graphs: {len(train_dataset)}")
    print(f"  Test graphs: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset


def demonstrate_batching(loader):
    """
    Show how PyTorch Geometric batches multiple graphs.
    
    Args:
        loader: A DataLoader
    """
    batch = next(iter(loader))
    
    print("=" * 60)
    print("Batching Demonstration")
    print("=" * 60)
    
    print(f"\nBatch object: {batch}")
    print(f"\nThe 'batch' tensor assigns each node to its graph:")
    print(f"  batch.batch: {batch.batch}")
    print(f"  Number of graphs in batch: {batch.num_graphs}")
    print(f"  Total nodes in batch: {batch.x.shape[0]}")
    print(f"  Total edges in batch: {batch.edge_index.shape[1]}")
    
    # Show how nodes are distributed
    nodes_per_graph = [(batch.batch == i).sum().item() for i in range(batch.num_graphs)]
    print(f"  Nodes per graph: {nodes_per_graph}")


if __name__ == "__main__":
    # Load dataset
    print("Loading MUTAG dataset...\n")
    dataset = load_mutag()
    
    # Show statistics
    stats = dataset_statistics(dataset)
    
    # Inspect a single graph
    print("\n")
    inspect_single_graph(dataset[0], idx=0)
    
    # Create data loaders
    print("\n")
    train_loader, test_loader, train_data, test_data = create_data_loaders(
        dataset, 
        train_ratio=0.8, 
        batch_size=32
    )
    
    # Demonstrate batching
    print("\n")
    demonstrate_batching(train_loader)
    
    print("\n" + "=" * 60)
    print("Data loading complete! Ready for GNN implementation.")
    print("=" * 60)