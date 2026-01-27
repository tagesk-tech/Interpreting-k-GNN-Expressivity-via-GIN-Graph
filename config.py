"""
config.py
Central configuration for the k-GNN + GIN-Graph interpretation project.

This file contains all hyperparameters and settings used across the project.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


# Dataset-specific configurations
DATASET_CONFIGS = {
    'mutag': {
        'max_nodes': 28,
        'num_node_features': 7,  # Atom types: C, N, O, F, I, Cl, Br
        'num_classes': 2,
        'gin_max_nodes': 28,  # Same as max_nodes for small graphs
    },
    'dd': {
        'max_nodes': 500,
        'num_node_features': 89,  # DD has 89 node features
        'num_classes': 2,
        'gin_max_nodes': 80,  # Smaller for GIN generation (memory efficiency)
    },
    'proteins': {
        'max_nodes': 620,
        'num_node_features': 3,  # PROTEINS has 3 node features
        'num_classes': 2,
        'gin_max_nodes': 50,  # Smaller for GIN generation (median graph ~26 nodes)
    },
}


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: str = "MUTAG"
    root: str = "./data"
    train_ratio: float = 0.8
    batch_size: int = 32
    seed: int = 42
    max_nodes: int = 28
    num_node_features: int = 7
    gin_max_nodes: int = 28  # Max nodes for GIN generation (can be smaller than max_nodes)

    @classmethod
    def from_dataset(cls, dataset_name: str, **kwargs) -> 'DataConfig':
        """
        Create a DataConfig from a dataset name with appropriate defaults.

        Args:
            dataset_name: Name of the dataset (mutag, dd, proteins)
            **kwargs: Override any default parameters

        Returns:
            DataConfig with dataset-specific settings
        """
        name_lower = dataset_name.lower()
        if name_lower not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_CONFIGS.keys())}"
            )

        config = DATASET_CONFIGS[name_lower]
        return cls(
            name=dataset_name.upper(),
            max_nodes=kwargs.get('max_nodes', config['max_nodes']),
            num_node_features=kwargs.get('num_node_features', config['num_node_features']),
            gin_max_nodes=kwargs.get('gin_max_nodes', config['gin_max_nodes']),
            root=kwargs.get('root', './data'),
            train_ratio=kwargs.get('train_ratio', 0.8),
            batch_size=kwargs.get('batch_size', 32),
            seed=kwargs.get('seed', 42),
        )


@dataclass
class KGNNConfig:
    """k-GNN model configuration."""
    hidden_dim: int = 64
    num_layers_1gnn: int = 3
    num_layers_2gnn: int = 2
    num_layers_3gnn: int = 2
    dropout: float = 0.5
    learning_rate: float = 0.01
    epochs: int = 100
    

@dataclass
class GINGraphConfig:
    """GIN-Graph generator/discriminator configuration."""
    latent_dim: int = 32
    hidden_dim: int = 128
    generator_dropout: float = 0.0
    learning_rate: float = 0.001
    epochs: int = 300
    batch_size: int = 64
    
    # Dynamic weighting scheme
    lambda_p: float = 0.4  # Percentage of training before increasing lambda
    lambda_k: float = 10.0  # Steepness of transition
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    
    # WGAN-GP
    gp_lambda: float = 10.0
    n_critic: int = 1  # Train discriminator n times per generator update
    
    # Gumbel-Softmax temperature
    temperature: float = 1.0
    eval_temperature: float = 0.1  # Lower temp for sharper discrete outputs


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    kgnn: KGNNConfig = field(default_factory=KGNNConfig)
    gin_graph: GINGraphConfig = field(default_factory=GINGraphConfig)
    
    # Models to train and interpret
    models: List[str] = field(default_factory=lambda: ['1gnn', '12gnn', '123gnn'])
    
    # Target class for interpretation (0=Mutagen, 1=Non-Mutagen)
    target_class: int = 0
    
    # Output directories
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    figures_dir: str = "./figures"
    
    # Device
    device: str = "auto"  # 'auto', 'cpu', 'cuda'
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


# Backward compatibility: MUTAG-specific constants
# For new code, use gin_handlers.get_handler(dataset).node_labels etc.
ATOM_LABELS = {
    0: 'C',   # Carbon
    1: 'N',   # Nitrogen
    2: 'O',   # Oxygen
    3: 'F',   # Fluorine
    4: 'I',   # Iodine
    5: 'Cl',  # Chlorine
    6: 'Br'   # Bromine
}

ATOM_COLORS = {
    'C': '#FFA500',    # Orange
    'N': '#00BFFF',    # Deep sky blue
    'O': '#FF0000',    # Red
    'F': '#32CD32',    # Lime green
    'I': '#800080',    # Purple
    'Cl': '#90EE90',   # Light green
    'Br': '#8B4513',   # Saddle brown
    '?': '#808080'     # Gray (unknown)
}

# Default class names (MUTAG)
CLASS_NAMES = {
    0: 'Mutagen',
    1: 'Non-Mutagen'
}


def get_class_names(dataset_name: str = 'mutag') -> dict:
    """
    Get class names for a dataset using the handler system.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary mapping class index to name
    """
    from gin_handlers import get_handler
    handler = get_handler(dataset_name)
    return handler.class_names


def get_class_name(class_idx: int, dataset_name: str = 'mutag') -> str:
    """
    Get class name for a specific class index.

    Args:
        class_idx: Class index
        dataset_name: Name of the dataset

    Returns:
        Class name string
    """
    names = get_class_names(dataset_name)
    return names.get(class_idx, f'Class {class_idx}')
