# Hierarchical k-GNN with GIN-based Interpretability

A PyTorch implementation of Hierarchical k-dimensional Graph Neural Networks with layer-wise interpretability analysis.

## Overview

This implementation is based on three key papers:

1. **"Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks"** (Morris et al., AAAI 2019)
   - Introduces k-GNNs that operate on k-element subsets of nodes
   - Shows k-GNNs have same expressiveness as k-WL graph isomorphism test
   - Proposes hierarchical 1-2-3-GNN architecture

2. **"How Powerful Are Graph Neural Networks?"** (Xu et al., ICLR 2019)
   - Proves GNNs are at most as powerful as 1-WL
   - Introduces Graph Isomorphism Network (GIN) as maximally expressive 1-GNN
   - Shows sum aggregation is more powerful than mean/max

3. **"GIN-Graph: A Generative Interpretation Network"** (Yue et al., 2025)
   - Model-level explanations for GNNs
   - Uses GANs to generate explanation graphs
   - Introduces validation score for filtering invalid explanations

## Key Concepts

### k-dimensional GNNs

Standard GNNs (1-GNNs) operate on nodes and aggregate neighbor features. They are limited by the 1-WL expressiveness bound - for example, they cannot:
- Count triangles
- Distinguish certain regular graphs
- Capture higher-order structures

k-GNNs overcome these limitations by:
- Operating on k-element subsets of nodes
- Performing message passing between subsets that differ by one element
- Capturing higher-order graph structures

### Hierarchical Architecture (1-2-3-GNN)

The hierarchical variant initializes each k-GNN using features from (k-1)-GNN:

```
f_k^(0)(s) = σ([f^iso(s), Σ_{u⊂s} f_{k-1}^(T)(u)] · W)
```

This creates a multi-scale representation of the graph.

### GIN Convolution

GIN uses sum aggregation with learnable ε:

```
h_v^(k) = MLP((1 + ε) · h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))
```

This is the most expressive aggregation under the 1-WL framework.

## Installation

```bash
# Install PyTorch first (follow pytorch.org instructions for your CUDA version)
pip install torch

# Install PyTorch Geometric
pip install torch-geometric

# Install additional dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from models import Hierarchical123GNN
from training import GraphDatasetLoader, HierarchicalKGNNTrainer

# Load dataset
loader = GraphDatasetLoader('./data')
dataset, num_features, num_classes = loader.load('MUTAG')

# Create model
model = Hierarchical123GNN(
    num_node_features=num_features,
    hidden_channels=64,
    num_classes=num_classes
)

# Train
trainer = HierarchicalKGNNTrainer(model, device='cuda')
results = trainer.cross_validate(dataset, num_folds=10, num_epochs=100)
print(f"Accuracy: {results['mean_test_acc']:.4f} ± {results['std_test_acc']:.4f}")
```

### Layer-wise Interpretability Analysis

```python
from interpretability import KLevelInterpretabilityReport, ExpressivenessAnalyzer

# Generate comprehensive report
report_gen = KLevelInterpretabilityReport(model, test_data)
report = report_gen.generate_report()
report_gen.print_report(report)

# Analyze expressiveness at each k level
analyzer = ExpressivenessAnalyzer(model)
expressiveness = analyzer.measure_layer_expressiveness(test_data)
```

### Testing Structure Distinguishability

```python
from interpretability import GraphStructureDistinguisher

distinguisher = GraphStructureDistinguisher()

# Create graph pairs that require different k to distinguish
g1, g2 = distinguisher.create_regular_graphs()  # 1-WL fails
results = distinguisher.test_distinguishability(model, g1, g2)
print(results)  # Shows which k levels can distinguish
```

### Model-Level Explanations

```python
from interpretability import GINGraphExplainer

explainer = GINGraphExplainer(model, hidden_dim=32, num_nodes=10)

# Generate explanation for class 0
best_adj, best_features = explainer.generate_explanation(
    target_class=0,
    num_iterations=1000
)
```

## Running Experiments

```bash
# Run all experiments
python experiments/main_experiment.py --experiment all

# Run specific experiment
python experiments/main_experiment.py --experiment expressiveness
python experiments/main_experiment.py --experiment layer_analysis
python experiments/main_experiment.py --experiment distinguishability
python experiments/main_experiment.py --experiment benchmark
python experiments/main_experiment.py --experiment explanation

# With custom parameters
python experiments/main_experiment.py \
    --dataset MUTAG \
    --hidden_dim 64 \
    --epochs 100 \
    --lr 0.01 \
    --batch_size 32
```

## Project Structure

```
hierarchical_kgnn/
├── models/
│   ├── __init__.py
│   └── kgnn.py              # k-GNN and GIN implementations
├── interpretability/
│   ├── __init__.py
│   └── layer_analysis.py    # Interpretability tools
├── training/
│   ├── __init__.py
│   └── trainer.py           # Training utilities
├── experiments/
│   ├── __init__.py
│   └── main_experiment.py   # Main experiment script
├── __init__.py
├── requirements.txt
└── README.md
```

## Key Classes

### Models

- **`GINConv`**: Graph Isomorphism Network convolution layer
- **`OneGNN`**: Standard 1-dimensional GNN using GIN convolutions
- **`KGNN`**: k-dimensional GNN operating on k-element subsets
- **`HierarchicalKGNN`**: Hierarchical architecture combining multiple k levels
- **`Hierarchical123GNN`**: Convenience class for 1-2-3-GNN

### Interpretability

- **`LayerEmbeddingAnalyzer`**: Extract and analyze embeddings from each layer
- **`GraphStructureDistinguisher`**: Test which structures can be distinguished
- **`ExpressivenessAnalyzer`**: Measure expressiveness metrics at each k level
- **`GINGraphExplainer`**: Generate model-level explanation graphs
- **`KLevelInterpretabilityReport`**: Generate comprehensive interpretability reports

### Training

- **`GraphDatasetLoader`**: Load standard graph classification datasets
- **`HierarchicalKGNNTrainer`**: Training with layer monitoring
- **`create_synthetic_dataset`**: Generate synthetic datasets for testing

## Theoretical Background

### Expressiveness Hierarchy

```
1-WL ≡ 1-GNN < 2-WL ≡ 2-GNN < 3-WL ≡ 3-GNN < ...
```

Each increase in k strictly increases expressiveness but also increases computational cost:
- 1-GNN: O(|V| + |E|) per layer
- k-GNN: O(|V|^k) per layer (but can use local neighborhoods to reduce)

### What Each k Level Captures

- **k=1**: Node-level patterns, local neighborhoods
- **k=2**: Edge-level patterns, can count triangles, some regular graph distinction
- **k=3**: Higher-order patterns, can distinguish most practical graph structures

### Validation Score for Explanations

From GIN-Graph paper:
```
v = (s × p × d)^(1/3)
```
where:
- s: Embedding similarity to class average
- p: Prediction probability for target class  
- d: Degree score (Gaussian based on reference statistics)

## Citation

If you use this code, please cite the original papers:

```bibtex
@inproceedings{morris2019weisfeiler,
  title={Weisfeiler and leman go neural: Higher-order graph neural networks},
  author={Morris, Christopher and Ritzert, Martin and Fey, Matthias and Hamilton, William L and Lenssen, Jan Eric and Rattan, Gaurav and Grohe, Martin},
  booktitle={AAAI},
  year={2019}
}

@inproceedings{xu2019powerful,
  title={How powerful are graph neural networks?},
  author={Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  booktitle={ICLR},
  year={2019}
}

@article{yue2025gingraph,
  title={GIN-Graph: A Generative Interpretation Network for Model-Level Explanation of Graph Neural Networks},
  author={Yue, Xiao and Qu, Guangzhi and Gan, Lige},
  journal={arXiv preprint arXiv:2503.06352},
  year={2025}
}
```

## License

MIT License
