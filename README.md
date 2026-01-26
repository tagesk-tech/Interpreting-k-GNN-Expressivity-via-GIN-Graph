# k-GNN Interpretation with GIN-Graph

This project investigates whether higher-order Graph Neural Networks (k-GNNs) provide better structural interpretations compared to standard 1-GNNs. We use the GIN-Graph method for model-level explanation generation.

## Research Question

**Does using higher-order k-GNN architectures lead to better interpretable patterns when generating model-level explanations?**

We compare:
- **1-GNN**: Standard message passing on nodes
- **1-2-GNN**: Hierarchical model using pairs of nodes
- **1-2-3-GNN**: Full hierarchical model using triplets

## Project Structure

```
kgnn_interpretation/
├── config.py              # Central configuration
├── data_loader.py         # MUTAG dataset loading utilities
├── models_kgnn.py         # k-GNN model implementations
├── gin_generator.py       # GIN-Graph generator and discriminator
├── model_wrapper.py       # Wrapper for dense inputs
├── dynamic_weighting.py   # Dynamic loss weighting scheme
├── metrics.py             # Evaluation metrics for explanations
├── visualize.py           # Visualization utilities
├── train_kgnn.py          # k-GNN training script
├── train_gin_graph.py     # GIN-Graph training script
├── run_experiment.py      # Full experiment runner
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

### 3. Run Full Experiment

```bash
# Run complete experiment pipeline
python run_experiment.py

# Skip k-GNN training if models already exist
python run_experiment.py --skip_kgnn_training

# Run only specific models
python run_experiment.py --models 1gnn 123gnn
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

## Evaluation Metrics

### Validation Score
```
v = (s × p × d)^(1/3)
```
- **s**: Embedding similarity to class centroid
- **p**: Prediction probability for target class  
- **d**: Degree score (structural validity)

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

After running experiments:

```
results/experiment_TIMESTAMP/
├── experiment_report.json        # Full results summary
├── gin_graph_1gnn.pt            # Generator checkpoints
├── gin_graph_12gnn.pt
├── gin_graph_123gnn.pt
└── figures/
    ├── explanations_1gnn.png    # Best explanations per model
    ├── training_1gnn.png        # Training curves
    ├── metrics_1gnn.png         # Metric distributions
    ├── model_comparison.png     # Cross-model comparison
    └── atom_legend.png          # Atom type legend
```

## References

1. Morris, C., et al. (2019). "Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks." AAAI.

2. Yue, X., et al. (2025). "GIN-Graph: A Generative Interpretation Network for Model-Level Explanation of Graph Neural Networks."

## License

MIT License
