stuff to do:
1. ~~Run experiments across datasets~~ DONE (MUTAG: all 3 models, PROTEINS: 1gnn + 12gnn)
2. ~~analyze and compare the generated explanations~~ DONE (all results regenerated with corrected embedding similarity)
3. draw conclusions about whether higher-order k-GNNs produce better interpretations

remaining gaps:
- MUTAG 123gnn class0 GIN needs retraining (current checkpoint only 1 epoch)
- MUTAG 1gnn class0 GIN needs retraining (current checkpoint only 49 epochs)
- PROTEINS 12gnn GIN needs retraining (current checkpoint only 29 epochs)
- PROTEINS 123gnn: k-GNN not trained yet, no GIN experiments


# k-GNN Interpretation with GIN-Graph

This project investigates whether higher-order Graph Neural Networks (k-GNNs) provide better structural interpretations compared to standard 1-GNNs. We use the GIN-Graph method for model-level explanation generation.

## Research Question

**Does using higher-order k-GNN architectures lead to better interpretable patterns when generating model-level explanations?**

We compare:
- **1-GNN**: Standard message passing on nodes
- **1-2-GNN**: Hierarchical model using pairs of nodes
- **1-2-3-GNN**: Full hierarchical model using triplets

> **Note on standalone 2-GNN / 3-GNN**: These models are available for k-GNN training but are **on standby** for GIN-Graph explanation generation. The hierarchical variants (1-2-GNN, 1-2-3-GNN) are recommended because they include a 1-GNN component that provides differentiable gradient flow through the adjacency matrix during generation. Standalone 2-GNN/3-GNN rely entirely on k-set construction which is non-differentiable through the generator, causing GIN-Graph training to stagnate.

## Project Structure

```
k-GNN/
├── config.py              # Central configuration
├── data_loader.py         # Dataset loading utilities (MUTAG, PROTEINS, DD)
├── models_kgnn.py         # k-GNN model implementations
├── gin_generator.py       # GIN-Graph generator and discriminator
├── model_wrapper.py       # Dense differentiable wrapper for GIN-Graph
├── dynamic_weighting.py   # Dynamic loss weighting scheme
├── metrics.py             # Evaluation metrics for explanations
├── visualize.py           # Visualization utilities
├── train_kgnn.py          # k-GNN training script
├── train_gin_graph.py     # GIN-Graph training script
├── research.py            # Analysis script (no training, evaluation + figures)
├── gin_handlers/          # Dataset-specific visualization handlers
└── requirements.txt       # Dependencies
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train k-GNN Models

```bash
# Train all models
python train_kgnn.py --all --epochs 100

# Or train specific model
python train_kgnn.py --model 1gnn --epochs 100
python train_kgnn.py --model 12gnn --epochs 100
python train_kgnn.py --model 123gnn --epochs 100
```

### 2. Train GIN-Graph Generators

```bash
# Generate explanations for the Mutagen class (0)
python train_gin_graph.py --model 1gnn --target_class 0 --epochs 300
python train_gin_graph.py --model 12gnn --target_class 0 --epochs 300
python train_gin_graph.py --model 123gnn --target_class 0 --epochs 300
```

### 3. Analyze Results (no training)

```bash
# Evaluate k-GNN checkpoints on test set
python research.py gnn --model 1gnn --dataset mutag
python research.py gnn --model 12gnn --dataset mutag

# Generate explanations from GIN-Graph checkpoints
python research.py gin --model 1gnn --dataset mutag --target_class 0
python research.py gin --model 12gnn --dataset mutag --target_class 0 --num_samples 200
```

## Theoretical Background

### k-GNNs (Morris et al., 2019)

Standard GNNs have the same expressive power as the 1-dimensional Weisfeiler-Leman (1-WL) algorithm. k-GNNs generalize this by performing message passing on k-sets (subsets of k nodes):

- **1-GNN**: Message passing between individual nodes
- **2-GNN**: Message passing between pairs of nodes
- **3-GNN**: Message passing between triplets of nodes

Higher k values capture more complex structural patterns but at increased computational cost.

### GIN-Graph (Yue et al., 2025)

GIN-Graph generates model-level explanations using:
1. **Generator (G)**: Creates graph structures from noise
2. **Discriminator (D)**: Distinguishes real from generated graphs
3. **Pre-trained GNN**: Guides generation toward class-specific patterns

Key innovations:
- **Dynamic loss weighting**: Gradually shifts from realism (GAN loss) to class-specificity (GNN loss)
- **Validation score**: Combines prediction probability, embedding similarity, and structural validity

### Dense Wrapper for Differentiable GIN-Graph Training

A key challenge in applying GIN-Graph to higher-order k-GNNs is maintaining gradient flow from the pretrained classifier back to the generator. The `DenseToSparseWrapper` in `model_wrapper.py` provides:

| Component | Approach | Gradient flow |
|-----------|----------|--------------|
| **1-GNN** | Dense batched message passing: `σ(H·W1 + A·H·W2)` | Full (through adj) |
| **2-GNN** | Dense pair features with einsum aggregation | Full (through adj) |
| **3-GNN** | Sparse k-set construction with soft iso-types | Partial (features + soft iso) |

- **1-GNN + 2-GNN**: Fully differentiable — the continuous adjacency matrix from the generator flows through all operations, so the GNN guidance loss provides gradient for both node features and graph structure.
- **3-GNN**: Uses optimized sparse construction (`torch.no_grad()` for structure, soft iso-types for partial gradient). Full dense 3-set tensors `[B,N,N,N,D]` would exceed memory for most graph sizes.

## Evaluation Metrics

### Validation Score
```
v = (s × p × d)^(1/3)
```
- **s**: Embedding similarity — cosine similarity between the generated graph's embedding (from the dense wrapper) and the precomputed class centroid (mean embedding of real target-class graphs via the sparse k-GNN). Measures whether the explanation lives in the right region of embedding space.
- **p**: Prediction probability for target class
- **d**: Degree score (structural validity via Gaussian kernel centered at class mean degree)

> **Fixed (c4d9e91)**: Earlier versions used `s = p` as a simplification, collapsing the metric to `v = (p² × d)^(1/3)`. This made the score blind to embedding-space shortcuts where a graph could achieve high prediction confidence without being structurally similar to real class members. The fix computes actual cosine similarity using `get_embedding()` on both the sparse k-GNN (for the centroid) and the dense wrapper (for generated graphs).

### Granularity
```
k = 1 - min(1, num_nodes / avg_class_nodes)
```
- 0: Coarse-grained (full graph patterns)
- →1: Fine-grained (small substructures)

## Expected Results

### MUTAG Dataset

The MUTAG dataset contains 188 molecular graphs classified as mutagenic or non-mutagenic. Key structural features include:
- NO₂ groups (strong mutagenic indicator)
- NH₂ groups (also mutagenic)
- Carbon ring structures

### Datasets

Experiments are conducted on **MUTAG** and **PROTEINS**. The **DD** dataset is excluded from the current experiments due to its large graph sizes (up to 500 nodes) and high feature dimensionality (89 features), which make GIN-Graph training prohibitively slow and memory-intensive. DD support is retained in the codebase for future work.

### Hypothesis

Higher-order k-GNNs should produce explanations that:
1. Better capture multi-node patterns (e.g., functional groups)
2. Have higher validation scores
3. Show more consistent structural motifs

## Configuration

Edit `config.py` to adjust:
- Model hyperparameters
- Training epochs
- Batch sizes
- Dynamic weighting parameters

## Output Files

The project uses a 3-stage directory layout matching the pipeline:

```
checkpoints/                          ← Stage 1: Train k-GNNs
├── mutag_1gnn.pt
├── mutag_12gnn.pt
└── mutag_123gnn.pt

gin_checkpoints/                      ← Stage 2: Train GIN-Graph generators
├── mutag/
│   ├── 1gnn_class0.pt               # Final GIN-Graph models
│   ├── 12gnn_class0.pt
│   └── training/                     # Intermediate artifacts
│       ├── ckpt_mutag_12gnn_epoch0.pt
│       └── samples_mutag_12gnn_epoch0.npz
└── proteins/
    ├── 1gnn_class0.pt
    └── training/

results/                              ← Stage 3: Analysis results
├── mutag/
│   ├── 1gnn/report.json             # k-GNN evaluation
│   ├── 1gnn_class0/                 # GIN-Graph analysis per class
│   │   ├── figures/
│   │   ├── explanations.npz
│   │   └── report.json
│   └── 12gnn_class1/
└── proteins/
    └── ...
```

Intermediate `.npz` sample files can be visualized offline with `visualize_standalone.py`
(no torch required):

```bash
python visualize_standalone.py gin_checkpoints/mutag/training/samples_mutag_12gnn_epoch100.npz -o fig.png
```

## References

1. Morris, C., et al. (2019). "Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks." AAAI.

2. Yue, X., et al. (2025). "GIN-Graph: A Generative Interpretation Network for Model-Level Explanation of Graph Neural Networks."

## License

MIT License
