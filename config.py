"""
config.py
Central configuration for the k-GNN + GIN-Graph interpretation project.

This file contains all hyperparameters and settings used across the project.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: str = "MUTAG"
    root: str = "./data"
    train_ratio: float = 0.8
    batch_size: int = 32
    seed: int = 42
    max_nodes: int = 28  # Maximum nodes in MUTAG
    num_node_features: int = 7  # Atom types: C, N, O, F, I, Cl, Br


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


# Atom type mapping for visualization
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

CLASS_NAMES = {
    0: 'Mutagen',
    1: 'Non-Mutagen'
}
