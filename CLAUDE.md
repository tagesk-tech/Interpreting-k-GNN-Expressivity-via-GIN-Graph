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
# Models + intermediate samples → ./gin_checkpoints/, final analysis → ./results/
python train_gin_graph.py --dataset mutag --model 1gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset proteins --model 1gnn --target_class 0 --epochs 300
```

### Analysis (no training)
```bash
# Evaluate a k-GNN checkpoint on the test set
python research.py gnn --model 1gnn --dataset mutag

# Generate explanations from existing GIN-Graph checkpoint
python research.py gin --model 1gnn --dataset mutag --target_class 0
python research.py gin --model 123gnn --dataset mutag --target_class 0 --num_samples 200

# Outputs go to results/{dataset}/{model}/ or results/{dataset}/{model}_class{N}/
```

### Supported Datasets
| Dataset  | Graphs | Node Features | Max Nodes | GIN Gen Size | Description | Status |
|----------|--------|---------------|-----------|--------------|-------------|--------|
| MUTAG    | 188    | 7 (atoms)     | 28        | 28           | Mutagenic compounds | Active |
| PROTEINS | 1113   | 3 (secondary) | 620       | 50           | Protein structures | Active |
| DD       | 1178   | 89 (amino acids) | 500    | 80           | Large protein graphs | Excluded |

**Note**: `GIN Gen Size` is the max nodes for explanation generation. Large-graph datasets use smaller generation sizes for memory efficiency - explanations highlight key substructures, not full graphs.

**DD dataset excluded**: DD is excluded from the current experiments due to its large graph sizes (up to 500 nodes) and high feature dimensionality (89 features), which make GIN-Graph training prohibitively slow and memory-intensive. The codebase retains DD support for future work.

## Architecture

### Pipeline Flow (3-step checkpoint workflow)
```
1. train_kgnn.py      → checkpoints/{dataset}_{model}.pt
2. train_gin_graph.py  → gin_checkpoints/{dataset}/{model}_class{N}.pt
3. research.py         → results/{dataset}/{model}_class{N}/ (figures, reports)
```

### Core Components

**Models (`models_kgnn.py`)**: k-GNN implementations using k-set message passing
- `OneGNNLayer`: Standard node-level message passing
- `KSetLayer`: Generic layer for 2-GNN (node pairs) and 3-GNN (triplets)
- Supported models: **1-GNN, 1-2-GNN, 1-2-3-GNN** (hierarchical only)
- Standalone 2-GNN/3-GNN class definitions remain in the file but are removed from all factories, CLIs, and tests
- **k-Set Sampling**: For large graphs (PROTEINS, DD), k-sets are randomly sampled to limit memory/time:
  - `max_pairs=5000` for 2-sets (vs 192K pairs for 620-node graph)
  - `max_triplets=3000` for 3-sets (vs 39M triplets for 620-node graph)
  - Adjustable in `_build_2sets()` and `_build_3sets()` methods

**Generator (`gin_generator.py`)**: Creates graphs from noise using Gumbel-Softmax for differentiable discrete sampling
- `GINGenerator`: Outputs adjacency matrix and node features
- `GINDiscriminator`: WGAN-GP discriminator for realism validation

**Model Wrapper (`model_wrapper.py`)**: Differentiable bridge between dense generator output and pretrained k-GNN
- `DenseToSparseWrapper`: Uses dense forward passes for hierarchical models (1gnn, 12gnn, 123gnn)
  - **1-GNN**: Dense batched message passing `σ(H·W1 + A·H·W2)` — fully differentiable through adj
  - **2-GNN**: Dense pair features `[h_min || h_max || adj[i,j]]` + einsum aggregation — fully differentiable through adj
  - **3-GNN**: Optimized sparse with `torch.no_grad()` for structure + soft iso-types — partial gradient

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

### Output Directories (3-stage pipeline)
```
./checkpoints/              ← Stage 1: k-GNN models ({dataset}_{model}.pt)
./gin_checkpoints/{dataset}/ ← Stage 2: GIN-Graph models per dataset
./results/                  ← Stage 3: Final analysis, figures, reports
./data/                     ← Datasets (auto-downloaded)
```
- `./checkpoints/`: Trained k-GNN models (e.g., `mutag_1gnn.pt`)
- `./gin_checkpoints/{dataset}/`: GIN-Graph models per dataset (e.g., `gin_checkpoints/mutag/1gnn_class0.pt`); `training/` subfolder holds intermediate checkpoints (`ckpt_*.pt`) and samples (`samples_*.npz`)
- `./results/{dataset}/{model}/`: k-GNN evaluation from `research.py gnn` (report.json)
- `./results/{dataset}/{model}_class{N}/`: GIN-Graph analysis from `research.py gin` (figures/, explanations.npz, report.json)
