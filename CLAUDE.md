# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether higher-order Graph Neural Networks (k-GNNs) provide better structural interpretations compared to standard 1-GNNs using model-level explanation generation via the GIN-Graph method on the MUTAG dataset.

## Commands

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install torch torch-geometric torch-scatter torch-sparse
pip install -r requirements.txt
```

### Testing
```bash
python test_setup.py        # Quick setup verification
python test_integration.py  # Comprehensive integration tests (fast, no full training)
```

### Training
```bash
# Train k-GNN models (default: MUTAG)
python train_kgnn.py --all --epochs 100
python train_kgnn.py --model 1gnn --epochs 100
python train_kgnn.py --model 123gnn --epochs 100

# Train on different datasets (mutag, dd, proteins)
python train_kgnn.py --dataset mutag --model 1gnn --epochs 100
python train_kgnn.py --dataset dd --model 1gnn --epochs 100
python train_kgnn.py --dataset proteins --model 1gnn --epochs 100

# Train GIN-Graph generators (requires pre-trained k-GNN)
python train_gin_graph.py --dataset mutag --model 1gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset dd --model 1gnn --target_class 0 --epochs 300

# Full experiment pipeline (default: MUTAG)
python run_experiment.py                              # Run on MUTAG (default)
python run_experiment.py --dataset dd                 # Run on DD dataset
python run_experiment.py --dataset proteins           # Run on PROTEINS dataset
python run_experiment.py --skip_kgnn_training         # Use existing checkpoints
python run_experiment.py --models 1gnn 123gnn         # Specific models only
python run_experiment.py --dataset dd --models 1gnn   # Combine options
```

### Supported Datasets
| Dataset  | Graphs | Node Features | Max Nodes | GIN Gen Size | Description |
|----------|--------|---------------|-----------|--------------|-------------|
| MUTAG    | 188    | 7 (atoms)     | 28        | 28           | Mutagenic compounds |
| DD       | 1178   | 89 (amino acids) | 500    | 80           | Large protein graphs |
| PROTEINS | 1113   | 3 (secondary) | 620       | 50           | Protein structures |

**Note**: `GIN Gen Size` is the max nodes for explanation generation. Large-graph datasets use smaller generation sizes for memory efficiency - explanations highlight key substructures, not full graphs.

## Architecture

### Pipeline Flow
```
Dataset (MUTAG/DD/PROTEINS) → k-GNN Training → GIN-Graph Generation → Explanation Evaluation → Visualization
```

### Core Components

**Models (`models_kgnn.py`)**: k-GNN implementations using k-set message passing
- `OneGNNLayer`: Standard node-level message passing
- `KSetLayer`: Generic layer for 2-GNN (node pairs) and 3-GNN (triplets)
- Hierarchical models: 1-GNN, 2-GNN, 3-GNN, 1-2-GNN, 1-3-GNN, 1-2-3-GNN

**Generator (`gin_generator.py`)**: Creates graphs from noise using Gumbel-Softmax for differentiable discrete sampling
- `GINGenerator`: Outputs adjacency matrix and node features
- `GINDiscriminator`: WGAN-GP discriminator for realism validation

**Model Wrapper (`model_wrapper.py`)**: Critical bridge between dense generator output and sparse k-GNN input
- `DenseToSparseWrapper`: Maintains gradient flow through adjacency matrices

**Training (`train_gin_graph.py`)**: Joint optimization combining:
- WGAN-GP loss (graph realism)
- k-GNN guidance loss (class-specificity via embedding similarity + prediction confidence)
- Dynamic weighting (`dynamic_weighting.py`): Sigmoid schedule shifting from GAN to GNN optimization

**Evaluation (`metrics.py`)**: Validation score = (s × p × d)^(1/3)
- s: Embedding similarity to class centroid
- p: Target class prediction probability
- d: Degree score (structural validity via Gaussian kernel)

### Dataset Handlers (`gin_handlers/`)

Dataset-specific GIN handlers for proper visualization and labeling:
```
gin_handlers/
├── __init__.py       # Factory function: get_handler(dataset_name)
├── base.py           # DatasetHandler abstract base class
├── mutag/            # MUTAG: molecular graphs with atom types
│   └── handler.py    # 7 atom types: C, N, O, F, I, Cl, Br
├── proteins/         # PROTEINS: secondary structure types
│   └── handler.py    # 3 types: Helix, Sheet, Coil/Turn
└── dd/               # DD: amino acid types
    └── handler.py    # 89 features (20 amino acids + properties)
```

Usage:
```python
from gin_handlers import get_handler
handler = get_handler('proteins')  # or 'mutag', 'dd'
handler.plot_explanation_graph(adj, x, ax=ax)
handler.class_names  # {0: 'Non-Enzyme', 1: 'Enzyme'}
```

### Configuration

All hyperparameters are centralized in `config.py`:
- `DataConfig`: Dataset settings with `from_dataset()` factory for dataset-specific defaults
- `KGNNConfig`: Hidden dim, layers, dropout, learning rate
- `GINGraphConfig`: Latent dim, WGAN-GP settings, dynamic weighting params
- `get_class_name(class_idx, dataset)`: Get class label from handler

### Output Directories
- `./checkpoints/`: Saved k-GNN models (`{dataset}_{model}.pt`, e.g., `mutag_1gnn.pt`)
- `./results/experiment_TIMESTAMP/`: Full results including JSON reports and GIN-Graph checkpoints
- `./data/`: Datasets (auto-downloaded)
