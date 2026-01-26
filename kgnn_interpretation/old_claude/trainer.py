"""
Training utilities and dataset handling for Hierarchical k-GNN.

Includes:
- Data loading for standard graph classification benchmarks
- Training loops with layer-wise monitoring
- Evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader  # <--- Moved here in v2.0+
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import time


class GraphDatasetLoader:
    """
    Load and preprocess graph classification datasets.
    
    Supports TUDatasets: MUTAG, PROTEINS, PTC, NCI1, etc.
    """
    
    SUPPORTED_DATASETS = [
        'MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1', 'NCI109',
        'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB'
    ]
    
    def __init__(self, root: str = './data'):
        self.root = root
        
    def load(self, name: str, use_node_attr: bool = True) -> Tuple[List[Data], int, int]:
        """
        Load dataset by name.
        
        Returns:
            dataset: List of Data objects
            num_features: Number of node features
            num_classes: Number of classes
        """
        if name not in self.SUPPORTED_DATASETS:
            print(f"Warning: {name} not in tested datasets. Proceeding anyway.")
        
        dataset = TUDataset(root=self.root, name=name, use_node_attr=use_node_attr)
        
        # Handle datasets without node features
        if dataset.num_node_features == 0:
            # Use degree as node feature
            max_degree = 0
            for data in dataset:
                d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, d.max().item())
            
            # One-hot encode degrees
            processed_dataset = []
            for data in dataset:
                d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
                d = torch.clamp(d, max=max_degree)
                x = F.one_hot(d, num_classes=max_degree + 1).float()
                processed_data = Data(
                    x=x, edge_index=data.edge_index, y=data.y
                )
                processed_dataset.append(processed_data)
            
            return processed_dataset, max_degree + 1, dataset.num_classes
        
        return list(dataset), dataset.num_node_features, dataset.num_classes
    
    def get_dataset_stats(self, dataset: List[Data]) -> Dict:
        """Compute dataset statistics."""
        stats = {
            'num_graphs': len(dataset),
            'avg_nodes': np.mean([d.num_nodes for d in dataset]),
            'avg_edges': np.mean([d.edge_index.size(1) / 2 for d in dataset]),
            'max_nodes': max([d.num_nodes for d in dataset]),
            'min_nodes': min([d.num_nodes for d in dataset]),
        }
        
        if hasattr(dataset[0], 'y'):
            labels = [d.y.item() for d in dataset]
            stats['num_classes'] = len(set(labels))
            stats['class_distribution'] = {
                label: labels.count(label) for label in set(labels)
            }
        
        # Compute degree statistics for validation score
        all_degrees = []
        for data in dataset:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
            avg_degree = d.mean().item()
            all_degrees.append(avg_degree)
        
        stats['mean_avg_degree'] = np.mean(all_degrees)
        stats['std_avg_degree'] = np.std(all_degrees)
        
        return stats


class HierarchicalKGNNTrainer:
    """
    Training utilities for hierarchical k-GNN models.
    
    Includes layer-wise monitoring for interpretability.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'layer_stats': []
        }
        
    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, monitor_layers: bool = False) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        layer_activations = defaultdict(list)
        
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            if monitor_layers:
                out, layer_info = self.model(
                    batch.x, batch.edge_index, batch.batch, return_layer_info=True
                )
                # Store activation statistics
                for key, layers in layer_info.items():
                    for i, layer in enumerate(layers):
                        if len(layer) > 0:
                            activation = layer[-1]  # Final layer activation
                            layer_activations[f'{key}_layer_{i}'].append({
                                'mean': activation.mean().item(),
                                'std': activation.std().item(),
                                'max': activation.max().item()
                            })
            else:
                out = self.model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        if monitor_layers:
            # Aggregate layer statistics
            layer_stats = {}
            for key, stats_list in layer_activations.items():
                layer_stats[key] = {
                    'mean_activation': np.mean([s['mean'] for s in stats_list]),
                    'std_activation': np.mean([s['std'] for s in stats_list]),
                    'max_activation': np.max([s['max'] for s in stats_list])
                }
            self.training_history['layer_stats'].append(layer_stats)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
        
        return total_loss / total, correct / total
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100, learning_rate: float = 0.01,
              weight_decay: float = 0.0, patience: int = 20,
              monitor_layers: bool = True, verbose: bool = True) -> Dict:
        """
        Full training loop with optional layer monitoring.
        """
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, monitor_layers
            )
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            epoch_time = time.time() - start_time
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s) - "
                msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loader is not None:
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(msg)
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_val_acc': best_val_acc,
            'final_train_acc': self.training_history['train_acc'][-1],
            'num_epochs': len(self.training_history['train_loss'])
        }
    
    def cross_validate(self, dataset: List[Data], num_folds: int = 10,
                       batch_size: int = 32, num_epochs: int = 100,
                       **train_kwargs) -> Dict:
        """
        Perform k-fold cross-validation.
        """
        labels = [d.y.item() for d in dataset]
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
            print(f"\nFold {fold + 1}/{num_folds}")
            
            # Reset model weights
            self._reset_model()
            
            # Create data loaders
            train_data = [dataset[i] for i in train_idx]
            test_data = [dataset[i] for i in test_idx]
            
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size)
            
            # Train
            train_result = self.train(
                train_loader, test_loader, num_epochs=num_epochs, **train_kwargs
            )
            
            # Final evaluation
            criterion = nn.CrossEntropyLoss()
            _, test_acc = self.evaluate(test_loader, criterion)
            
            fold_results.append({
                'train_acc': train_result['final_train_acc'],
                'test_acc': test_acc
            })
            
            print(f"Fold {fold + 1} Test Accuracy: {test_acc:.4f}")
        
        # Aggregate results
        test_accs = [r['test_acc'] for r in fold_results]
        
        return {
            'mean_test_acc': np.mean(test_accs),
            'std_test_acc': np.std(test_accs),
            'fold_results': fold_results
        }
    
    def _reset_model(self):
        """Reset model weights."""
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def create_synthetic_dataset(dataset_type: str = 'motif', 
                            num_graphs: int = 500) -> List[Data]:
    """
    Create synthetic datasets for testing k-GNN expressiveness.
    
    Types:
    - 'motif': Graphs with different motifs (house, cycle, etc.)
    - 'regular': Regular graphs that require k>1 to distinguish
    - 'wl_hard': Graphs specifically designed to be hard for 1-WL
    """
    dataset = []
    
    if dataset_type == 'motif':
        motifs = {
            'triangle': _create_triangle,
            'square': _create_square,
            'house': _create_house,
            'star': _create_star
        }
        
        for i in range(num_graphs):
            motif_type = list(motifs.keys())[i % len(motifs)]
            label = list(motifs.keys()).index(motif_type)
            
            # Create base graph (random tree)
            base = _create_random_tree(np.random.randint(5, 10))
            
            # Attach motif
            motif = motifs[motif_type]()
            data = _attach_motif(base, motif)
            data.y = torch.tensor([label])
            
            dataset.append(data)
    
    elif dataset_type == 'regular':
        for i in range(num_graphs):
            if i % 2 == 0:
                # Create one type of regular graph
                data = _create_regular_graph_type1(np.random.randint(6, 12))
                data.y = torch.tensor([0])
            else:
                # Create another type
                data = _create_regular_graph_type2(np.random.randint(6, 12))
                data.y = torch.tensor([1])
            
            dataset.append(data)
    
    elif dataset_type == 'wl_hard':
        for i in range(num_graphs):
            n = np.random.randint(8, 15)
            if i % 2 == 0:
                data = _create_wl_hard_graph1(n)
                data.y = torch.tensor([0])
            else:
                data = _create_wl_hard_graph2(n)
                data.y = torch.tensor([1])
            
            dataset.append(data)
    
    return dataset


def _create_triangle() -> Data:
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    x = torch.ones(3, 1)
    return Data(x=x, edge_index=edge_index)


def _create_square() -> Data:
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                               [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    x = torch.ones(4, 1)
    return Data(x=x, edge_index=edge_index)


def _create_house() -> Data:
    # House shape: square with a triangle on top
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0, 2, 4, 3, 4],
        [1, 0, 2, 1, 3, 2, 0, 3, 4, 2, 4, 3]
    ], dtype=torch.long)
    x = torch.ones(5, 1)
    return Data(x=x, edge_index=edge_index)


def _create_star() -> Data:
    # 5-pointed star
    center = 0
    edges = []
    for i in range(1, 6):
        edges.extend([[center, i], [i, center]])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.ones(6, 1)
    return Data(x=x, edge_index=edge_index)


def _create_random_tree(n: int) -> Data:
    """Create a random tree with n nodes."""
    edges = []
    for i in range(1, n):
        parent = np.random.randint(0, i)
        edges.extend([[parent, i], [i, parent]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty(2, 0, dtype=torch.long)
    x = torch.ones(n, 1)
    return Data(x=x, edge_index=edge_index)


def _attach_motif(base: Data, motif: Data) -> Data:
    """Attach a motif to a base graph."""
    base_nodes = base.num_nodes
    motif_nodes = motif.num_nodes
    
    # Offset motif edges
    motif_edges = motif.edge_index + base_nodes
    
    # Connect motif to random node in base
    connect_node = np.random.randint(0, base_nodes)
    connect_edge = torch.tensor([[connect_node, base_nodes],
                                 [base_nodes, connect_node]], dtype=torch.long)
    
    # Combine
    edge_index = torch.cat([base.edge_index, motif_edges, connect_edge], dim=1)
    x = torch.ones(base_nodes + motif_nodes, 1)
    
    return Data(x=x, edge_index=edge_index)


def _create_regular_graph_type1(n: int) -> Data:
    """Create a cycle with chords."""
    edges = []
    # Create cycle
    for i in range(n):
        edges.extend([[i, (i + 1) % n], [(i + 1) % n, i]])
    # Add some chords
    for i in range(0, n, 2):
        j = (i + n // 2) % n
        edges.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.ones(n, 1)
    return Data(x=x, edge_index=edge_index)


def _create_regular_graph_type2(n: int) -> Data:
    """Create a different regular graph."""
    edges = []
    # Create cycle
    for i in range(n):
        edges.extend([[i, (i + 1) % n], [(i + 1) % n, i]])
    # Add different pattern of chords
    for i in range(n // 3):
        j = (i + n // 3) % n
        edges.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.ones(n, 1)
    return Data(x=x, edge_index=edge_index)


def _create_wl_hard_graph1(n: int) -> Data:
    """Create a graph that's hard for 1-WL."""
    # Two cliques connected by a matching
    k = n // 2
    edges = []
    
    # First clique
    for i in range(k):
        for j in range(i + 1, k):
            edges.extend([[i, j], [j, i]])
    
    # Second clique
    for i in range(k, n):
        for j in range(i + 1, n):
            edges.extend([[i, j], [j, i]])
    
    # Connect cliques
    for i in range(min(k, n - k)):
        edges.extend([[i, k + i], [k + i, i]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.ones(n, 1)
    return Data(x=x, edge_index=edge_index)


def _create_wl_hard_graph2(n: int) -> Data:
    """Create another graph that 1-WL conflates with type 1."""
    k = n // 2
    edges = []
    
    # First clique
    for i in range(k):
        for j in range(i + 1, k):
            edges.extend([[i, j], [j, i]])
    
    # Second clique
    for i in range(k, n):
        for j in range(i + 1, n):
            edges.extend([[i, j], [j, i]])
    
    # Different connection pattern
    edges.extend([[0, k], [k, 0]])
    edges.extend([[k - 1, n - 1], [n - 1, k - 1]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.ones(n, 1)
    return Data(x=x, edge_index=edge_index)
