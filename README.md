Notes till Max:
1. Det verkar som att 12-modeler förstår sig mer på grafer än 1-gnn pga att den gissar fortfarande rätt men graferna är betydligt mer homogena vilket förmodligen betyder att den inte tittar på själva embeddingen utan snarare blir mer påverkad av message passing mellan noderna. 1-gnns har mycket mer sprid användning av själva atomerna vilket betyder att den tittar nog mer på vad som faktiskt är i varje nod. Jag skulle kolla här och se om man kan verifiera den här hypotesen

att göra Max: 
1. Förstå kopplinen mellan datasetet och output från GIN. Titta på flera grafer som genereras kanske bygg ut ett nytt dataset producerat från träningen och analysera dem graferna.
       1. Det här betyder att man analyserar de graferna som finns i mutag / proteins och jämför dem med de som genereras i slutet av pipen
       2. Tittar på kopplingar med vald model och hur de speglar output i det här fallet
3. Målet är enkelt: förstår sig k-GNN sig på grafiska strukturer bättre än 1-GNNs och kan man se det genom de producerade graferna av vår GIN
       1. Här kan du träna flera gånger med olika vilkor olika epocher etc var bara nogran och dokumnetera.



# k-GNN Interpretation with GIN-Graph

This project investigates whether a hierarchical 1-2-GNN provides better structural interpretations compared to a standard 1-GNN. We use the GIN-Graph method for model-level explanation generation and compare the two architectures across the MUTAG and PROTEINS datasets.

## Research Question

**Does adding 2-set (node-pair) message passing on top of a standard GNN lead to more interpretable model-level explanations?**

We compare:
- **1-GNN**: Standard message passing on nodes (1-WL equivalent)
- **1-2-GNN**: Hierarchical model — runs 1-GNN first, then 2-GNN on node pairs using the learned embeddings

The 1-2-GNN enriches the graph representation with pairwise structural information (which pairs of nodes are connected, how pairs relate to neighboring pairs) while keeping the 1-GNN component for differentiable gradient flow during GIN-Graph training.

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
├── compare_datasets.py    # Generated vs real dataset comparison
├── benchmark_kgnn.py      # Computational cost benchmarking
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
# Train the two models in the study
python train_kgnn.py --model 1gnn --epochs 100
python train_kgnn.py --model 12gnn --epochs 100

# On PROTEINS
python train_kgnn.py --dataset proteins --model 1gnn --epochs 100
python train_kgnn.py --dataset proteins --model 12gnn --epochs 100
```

### 2. Train GIN-Graph Generators

```bash
# MUTAG — both classes for each model
python train_gin_graph.py --dataset mutag --model 1gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset mutag --model 1gnn --target_class 1 --epochs 300
python train_gin_graph.py --dataset mutag --model 12gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset mutag --model 12gnn --target_class 1 --epochs 300

# PROTEINS — both classes for each model
python train_gin_graph.py --dataset proteins --model 1gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset proteins --model 1gnn --target_class 1 --epochs 300
python train_gin_graph.py --dataset proteins --model 12gnn --target_class 0 --epochs 300
python train_gin_graph.py --dataset proteins --model 12gnn --target_class 1 --epochs 300
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

### 4. Compare Generated vs Real Datasets

Generate large synthetic datasets from the trained GIN-Graph models and compare them against the original real data:

```bash
python compare_datasets.py --dataset mutag --num_samples 500
python compare_datasets.py --dataset proteins --num_samples 500
```

This generates 500 graphs per class from each model's GIN-Graph checkpoints, then runs three analyses:

- **Structural fidelity** — Compares degree distributions, graph sizes, and node type frequencies between generated and real graphs using KS tests
- **Cross-model classification** — Feeds each model's generated graphs through BOTH k-GNN classifiers to test whether the graphs are model-specific or universally recognized
- **Embedding space visualization** — t-SNE projections showing where generated graphs land relative to real data in embedding space

Output goes to `results/{dataset}/comparison/` with figures, a summary `report.json`, and the generated datasets as `.npz` files.

## Theoretical Background

### k-GNNs (Morris et al., 2019)

Standard GNNs have the same expressive power as the 1-dimensional Weisfeiler-Leman (1-WL) algorithm. k-GNNs generalize this by performing message passing on k-sets (subsets of k nodes):

- **1-GNN**: Message passing between individual nodes
- **2-GNN**: Message passing between pairs of nodes

Higher k values capture more complex structural patterns but at increased computational cost. Our study focuses on the step from k=1 to k=2, which is the most practical increase in expressiveness.

### Computational Cost

The 2-GNN component adds significant computational overhead due to O(n^2) pair construction:

| Operation | MUTAG 1-GNN | MUTAG 1-2-GNN | PROTEINS 1-GNN | PROTEINS 1-2-GNN |
|-----------|-------------|---------------|----------------|------------------|
| k-GNN forward pass | 1.9 ms | 99.5 ms (52x) | 6.8 ms | 2,214 ms (326x) |
| GIN-Graph generation (1 graph) | 0.02 s | 1.48 s (74x) | 0.04 s | 1.43 s (36x) |

On PROTEINS (50-node generation size), k-set construction alone takes ~2.2 seconds per forward pass. This makes the 1-2-GNN's GIN-Graph training roughly two orders of magnitude slower than the 1-GNN's.

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

Both the 1-GNN and 2-GNN components are fully differentiable — the continuous adjacency matrix from the generator flows through all operations, so the GNN guidance loss provides gradient for both node features and graph structure.

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

## Datasets

Experiments are conducted on **MUTAG** and **PROTEINS**.

### MUTAG

The MUTAG dataset contains 188 molecular graphs classified as mutagenic or non-mutagenic. Key structural features include:
- NO2 groups (strong mutagenic indicator)
- NH2 groups (also mutagenic)
- Carbon ring structures

### PROTEINS

The PROTEINS dataset contains 1113 protein graphs classified as enzyme or non-enzyme. Node features represent secondary structure types (Helix, Sheet, Coil/Turn).

### Hypothesis

The 1-2-GNN should produce explanations that:
1. Better capture multi-node patterns (e.g., functional groups, structural motifs)
2. Have higher validation scores due to richer structural representation
3. Show more consistent structural motifs across generated samples

## Configuration

Edit `config.py` to adjust:
- Model hyperparameters
- Training epochs
- Batch sizes
- Dynamic weighting parameters

## Output Files

The project uses a 3-stage directory layout matching the pipeline:

```
checkpoints/                          <- Stage 1: Train k-GNNs
├── mutag_1gnn.pt
└── mutag_12gnn.pt

gin_checkpoints/                      <- Stage 2: Train GIN-Graph generators
├── mutag/
│   ├── 1gnn_class0.pt               # Final GIN-Graph models
│   ├── 12gnn_class0.pt
│   └── training/                     # Intermediate artifacts
│       ├── ckpt_mutag_12gnn_epoch0.pt
│       └── samples_mutag_12gnn_epoch0.npz
└── proteins/
    ├── 1gnn_class0.pt
    └── training/

results/                              <- Stage 3 & 4: Analysis results
├── mutag/
│   ├── 1gnn/report.json             # k-GNN evaluation
│   ├── 1gnn_class0/                 # GIN-Graph analysis per class
│   │   ├── figures/
│   │   ├── explanations.npz
│   │   └── report.json
│   ├── 12gnn_class1/
│   └── comparison/                   # Generated vs real comparison
│       ├── figures/                  # Structural, cross-classification, t-SNE
│       ├── 1gnn_generated.npz       # 500+500 generated graphs
│       ├── 12gnn_generated.npz
│       └── report.json
└── proteins/
    └── ...
```

Intermediate `.npz` sample files can be visualized offline with `visualize_standalone.py`
(no torch required):

```bash
python visualize_standalone.py gin_checkpoints/mutag/training/samples_mutag_12gnn_epoch100.npz -o fig.png
```

## Extended Model Support

The codebase also supports **1-2-3-GNN** (hierarchical triplet model) and the **DD** dataset, though these are outside the scope of the current study:

- **1-2-3-GNN**: Adds 3-set (node triplet) message passing. Uses optimized sparse construction with `torch.no_grad()` for structure and soft iso-types for partial gradient flow. Available via `--model 123gnn`.
- **DD dataset**: 1178 large protein graphs (up to 500 nodes, 89 features). Excluded due to memory/time constraints for GIN-Graph training. Available via `--dataset dd`.
- **Standalone 2-GNN / 3-GNN**: Class definitions remain in `models_kgnn.py` but are removed from the factory, CLIs, and tests. These lack the 1-GNN component needed for differentiable GIN-Graph training.

## Key Findings

### Cross-Model Classification

When we generate 500 graphs per class from each model's GIN-Graph generators and classify them with *both* k-GNN models, a clear asymmetry emerges:

**MUTAG:**
| Generator | Same-model | Cross-model |
|-----------|-----------|-------------|
| 1-GNN class 0 | 100% | 90.4% |
| 1-GNN class 1 | 99.8% | 91.6% |
| 1-2-GNN class 0 | 99.8% | 100% |
| 1-2-GNN class 1 | 100% | 12.4% |

**PROTEINS:**
| Generator | Same-model | Cross-model |
|-----------|-----------|-------------|
| 1-GNN class 0 | 100% | 9.4% |
| 1-GNN class 1 | 99.8% | 99.6% |
| 1-2-GNN class 0 | 98.4% | 100% |
| 1-2-GNN class 1 | 100% | 98.0% |

Key observations:
- **1-GNN generators** produce graphs that are generally recognized by both models, especially for class 1 (high cross-model agreement).
- **1-2-GNN generators** show class-dependent behavior: class 0 graphs transfer well (100% cross on both datasets), but class 1 graphs on MUTAG transfer poorly (12.4%), suggesting the 1-2-GNN learns pairwise structural patterns invisible to standard message passing.
- **Asymmetric rejection on PROTEINS**: 1-GNN class-0 graphs are rejected by the 1-2-GNN (9.4% cross), while 1-2-GNN class-0 graphs are accepted by both (98.4%/100%). The 1-2-GNN's richer feature space enables it to distinguish patterns that the 1-GNN conflates.

### Structural Fidelity

Both models preserve mean degree well (within 1-2% of real data). Graph size regularization (added in the latest training round) brings generated sizes close to real class means:

| Dataset | Model | Class | Real Size | Gen Size | Real Degree | Gen Degree |
|---------|-------|-------|-----------|----------|-------------|------------|
| MUTAG | 1-GNN | 0 | 13.9 | 23.8 | 2.09 | 2.06 |
| MUTAG | 1-GNN | 1 | 19.8 | 27.3 | 2.24 | 2.23 |
| PROTEINS | 1-GNN | 0 | 48.7 | 46.6 | 3.80 | 3.81 |
| PROTEINS | 1-GNN | 1 | 23.7 | 24.0 | 3.64 | 3.73 |
| PROTEINS | 1-2-GNN | 0 | 48.7 | 50.0 | 3.80 | 3.81 |
| PROTEINS | 1-2-GNN | 1 | 23.7 | 29.5 | 3.64 | 3.64 |

PROTEINS sizes are now much closer to real means (previously ~42-44 nodes for class 1). MUTAG sizes still overshoot because the generator's fixed 28×28 output nearly matches max graph size, leaving less room for size regularization to act.

### Training Improvements

Three improvements were added to the GIN-Graph training pipeline:

1. **Graph size regularization** (`size_lambda=10.0`): Differentiable penalty on active node count deviation from class mean, using a sigmoid approximation of node activity.
2. **Node type distribution regularization** (`node_type_lambda=1.0`): L2 penalty on generated node type frequencies vs real class frequencies.
3. **Best checkpoint selection**: Training now tracks validation score at checkpoint intervals and saves the best model separately, rather than always using the last epoch.

## References

1. Morris, C., et al. (2019). "Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks." AAAI.

2. Yue, X., et al. (2025). "GIN-Graph: A Generative Interpretation Network for Model-Level Explanation of Graph Neural Networks."

## License

MIT License
