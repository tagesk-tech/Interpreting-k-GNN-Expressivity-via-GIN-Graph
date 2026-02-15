# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether a hierarchical 1-2-GNN provides better structural interpretations compared to a standard 1-GNN using model-level explanation generation via the GIN-Graph method. We compare the two architectures across the MUTAG and PROTEINS datasets.

## Key Fixes Applied

### Embedding similarity (s) metric — fixed in c4d9e91
The validation score `v = (s * p * d)^(1/3)` originally used `s = p` (prediction probability as a proxy for embedding similarity), collapsing the metric to `(p^2 * d)^(1/3)`. Fixed by computing actual cosine similarity between generated graph embeddings (via `DenseToSparseWrapper.get_embedding()`) and a precomputed class centroid (mean embedding of real target-class graphs via the sparse k-GNN). The centroid is saved in GIN-Graph checkpoints. All results have been regenerated with the corrected metric.

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
# Train the two models in the study
python train_kgnn.py --model 1gnn --epochs 100
python train_kgnn.py --model 12gnn --epochs 100

# On PROTEINS
python train_kgnn.py --dataset proteins --model 1gnn --epochs 100
python train_kgnn.py --dataset proteins --model 12gnn --epochs 100

# Train GIN-Graph generators (requires pre-trained k-GNN)
# Models + intermediate samples → ./gin_checkpoints/, final analysis → ./results/
python train_gin_graph.py --dataset mutag --model 1gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset mutag --model 12gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset proteins --model 1gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset proteins --model 12gnn --target_class 0 --epochs 300
```

### Analysis (no training)
```bash
# Evaluate k-GNN checkpoints on test set
python research.py gnn --model 1gnn --dataset mutag
python research.py gnn --model 12gnn --dataset mutag

# Generate explanations from existing GIN-Graph checkpoint
python research.py gin --model 1gnn --dataset mutag --target_class 0
python research.py gin --model 12gnn --dataset mutag --target_class 0 --num_samples 200

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

### Current Experiment Status
All experiments complete. All GIN-Graph models trained for 300 epochs. Results regenerated 2026-02-15.

| Dataset  | Model   | k-GNN | GIN c0 | GIN c1 | Analysis |
|----------|---------|-------|--------|--------|----------|
| MUTAG    | 1-GNN   | 89.5% | 300ep | 300ep | done |
| MUTAG    | 1-2-GNN | 89.5% | 300ep | 300ep | done |
| PROTEINS | 1-GNN   | 77.6% | 300ep | 300ep | done |
| PROTEINS | 1-2-GNN | 68.6% | 300ep | 300ep | done |

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
- Study models: **1-GNN** (standard) and **1-2-GNN** (hierarchical with pairwise message passing)
- Extended: 1-2-3-GNN also available via `--model 123gnn` (see Extended Model Support below)
- Standalone 2-GNN/3-GNN class definitions remain in the file but are removed from all factories, CLIs, and tests
- **k-Set Sampling**: For large graphs (PROTEINS, DD), k-sets are randomly sampled to limit memory/time:
  - `max_pairs=5000` for 2-sets (vs 192K pairs for 620-node graph)
  - Adjustable in `_build_2sets()` method

**Generator (`gin_generator.py`)**: Creates graphs from noise using Gumbel-Softmax for differentiable discrete sampling
- `GINGenerator`: Outputs adjacency matrix and node features
- `GINDiscriminator`: WGAN-GP discriminator for realism validation

**Model Wrapper (`model_wrapper.py`)**: Differentiable bridge between dense generator output and pretrained k-GNN
- `DenseToSparseWrapper`: Uses dense forward passes for hierarchical models
  - **1-GNN**: Dense batched message passing `σ(H·W1 + A·H·W2)` — fully differentiable through adj
  - **2-GNN**: Dense pair features `[h_min || h_max || adj[i,j]]` + einsum aggregation — fully differentiable through adj
  - Both components are fully differentiable — the continuous adjacency matrix from the generator flows through all operations

**Training (`train_gin_graph.py`)**: Joint optimization combining:
- WGAN-GP loss (graph realism)
- k-GNN guidance loss (class-specificity via embedding similarity + prediction confidence)
- Dynamic weighting (`dynamic_weighting.py`): Sigmoid schedule shifting from GAN to GNN optimization

**Evaluation (`metrics.py`)**: Validation score = (s × p × d)^(1/3)
- s: Cosine similarity between generated embedding (dense wrapper) and class centroid (sparse k-GNN on real graphs)
- p: Target class prediction probability
- d: Degree score (structural validity via Gaussian kernel)
- Class centroid is computed once per training run via `compute_class_centroid()` and saved in checkpoints

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

## Extended Model Support

The codebase also supports functionality beyond the current 1-GNN vs 1-2-GNN study:

- **1-2-3-GNN**: Adds 3-set (node triplet) message passing. Uses optimized sparse construction with `torch.no_grad()` for structure and soft iso-types for partial gradient flow. Available via `--model 123gnn`. Dense wrapper has partial gradient (3-GNN component uses sparse path).
- **DD dataset**: 1178 large protein graphs (up to 500 nodes, 89 features). Excluded due to memory/time constraints for GIN-Graph training. Available via `--dataset dd`.
- **Standalone 2-GNN / 3-GNN**: Class definitions remain in `models_kgnn.py` but are removed from the factory, CLIs, and tests. These lack the 1-GNN component needed for differentiable GIN-Graph training.
- **k-Set Sampling for 3-sets**: `max_triplets=3000` in `_build_3sets()` for memory-constrained 3-GNN training.
