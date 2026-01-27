"""
visualize.py
Visualization utilities for k-GNN and GIN-Graph results.

Uses dataset-specific handlers from gin_handlers for proper
node labeling and coloring.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional, Dict, Tuple
import os

from gin_handlers import get_handler, DatasetHandler


def get_default_handler() -> DatasetHandler:
    """Get the default handler (MUTAG for backward compatibility)."""
    return get_handler('mutag')


def adj_to_networkx(
    adj: np.ndarray,
    x: np.ndarray,
    edge_threshold: float = 0.5,
    remove_isolated: bool = True,
    dataset: str = 'mutag'
) -> Tuple[nx.Graph, Dict[int, str], List[str]]:
    """
    Convert adjacency matrix and features to NetworkX graph.

    Args:
        adj: Adjacency matrix [N, N]
        x: Node features [N, D] (one-hot)
        edge_threshold: Threshold for edge existence
        remove_isolated: Whether to remove isolated nodes
        dataset: Dataset name for proper labeling

    Returns:
        G: NetworkX graph
        node_labels: Dict mapping node index to label
        node_colors: List of colors for each node
    """
    handler = get_handler(dataset)
    return handler.adj_to_networkx(adj, x, edge_threshold, remove_isolated)


def plot_explanation_graph(
    adj: np.ndarray,
    x: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    edge_threshold: float = 0.5,
    node_size: int = 400,
    font_size: int = 10,
    seed: int = 42,
    dataset: str = 'mutag'
):
    """
    Plot a single explanation graph.

    Args:
        adj: Adjacency matrix [N, N]
        x: Node features [N, D]
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        edge_threshold: Threshold for edges
        node_size: Size of nodes
        font_size: Label font size
        seed: Random seed for layout
        dataset: Dataset name for proper labeling
    """
    handler = get_handler(dataset)
    handler.plot_explanation_graph(
        adj, x, ax=ax, title=title, edge_threshold=edge_threshold,
        node_size=node_size, font_size=font_size, seed=seed
    )


def plot_explanation_grid(
    adjs: np.ndarray,
    xs: np.ndarray,
    metrics: List = None,
    num_cols: int = 5,
    figsize: Tuple[int, int] = None,
    save_path: str = None,
    title: str = None,
    dataset: str = 'mutag'
):
    """
    Plot a grid of explanation graphs.

    Args:
        adjs: Adjacency matrices [batch, N, N]
        xs: Node features [batch, N, D]
        metrics: List of ExplanationMetrics (optional)
        num_cols: Number of columns in grid
        figsize: Figure size
        save_path: Path to save figure
        title: Overall title
        dataset: Dataset name for proper labeling
    """
    handler = get_handler(dataset)
    return handler.plot_explanation_grid(
        adjs, xs, metrics=metrics, num_cols=num_cols,
        figsize=figsize, save_path=save_path, title=title
    )


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None
):
    """
    Plot training history curves.

    Args:
        history: Dictionary with training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # D and G losses
    ax = axes[0, 0]
    ax.plot(history['d_loss'], label='Discriminator', alpha=0.7)
    ax.plot(history['g_loss'], label='Generator', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # GAN vs GNN loss
    ax = axes[0, 1]
    ax.plot(history['gan_loss'], label='GAN Loss', alpha=0.7)
    ax.plot(history['gnn_loss'], label='GNN Loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Generator Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lambda schedule
    ax = axes[1, 0]
    ax.plot(history['lambda'], color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Lambda')
    ax.set_title('Dynamic Weight Schedule')
    ax.grid(True, alpha=0.3)

    # Prediction probability
    ax = axes[1, 1]
    ax.plot(history['pred_prob'], color='purple', alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Prediction Probability')
    ax.set_title('Target Class Prediction Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")

    return fig


def plot_validation_scores_distribution(
    metrics: List,
    save_path: str = None
):
    """
    Plot distribution of validation metrics.

    Args:
        metrics: List of ExplanationMetrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    scores = [m.validation_score for m in metrics]
    probs = [m.prediction_probability for m in metrics]
    degrees = [m.degree_score for m in metrics]
    valid = [m.is_valid for m in metrics]

    # Validation scores
    ax = axes[0, 0]
    ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(scores), color='r', linestyle='--',
               label=f'Mean: {np.mean(scores):.3f}')
    ax.set_xlabel('Validation Score')
    ax.set_ylabel('Count')
    ax.set_title('Validation Score Distribution')
    ax.legend()

    # Prediction probabilities
    ax = axes[0, 1]
    ax.hist(probs, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(x=np.mean(probs), color='r', linestyle='--',
               label=f'Mean: {np.mean(probs):.3f}')
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()

    # Degree scores
    ax = axes[1, 0]
    ax.hist(degrees, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=np.mean(degrees), color='r', linestyle='--',
               label=f'Mean: {np.mean(degrees):.3f}')
    ax.set_xlabel('Degree Score')
    ax.set_ylabel('Count')
    ax.set_title('Degree Score Distribution')
    ax.legend()

    # Valid vs Invalid
    ax = axes[1, 1]
    valid_count = sum(valid)
    invalid_count = len(valid) - valid_count
    ax.bar(['Valid', 'Invalid'], [valid_count, invalid_count],
           color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Validity: {valid_count}/{len(valid)} '
                 f'({100*valid_count/len(valid):.1f}%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics distribution saved to: {save_path}")

    return fig


def create_comparison_figure(
    results: Dict[str, Dict],
    save_path: str = None,
    dataset: str = 'mutag'
):
    """
    Create comparison figure across different k-GNN models.

    Args:
        results: Dict of model_name -> {adjs, xs, metrics, ...}
        save_path: Path to save figure
        dataset: Dataset name for proper labeling
    """
    handler = get_handler(dataset)
    num_models = len(results)
    fig, axes = plt.subplots(num_models, 5, figsize=(15, 3 * num_models))

    if num_models == 1:
        axes = axes.reshape(1, -1)

    for row, (model_name, data) in enumerate(results.items()):
        adjs = data['adjs']
        xs = data['xs']
        metrics = data['metrics']

        # Get top 5 explanations
        indexed = list(enumerate(metrics))
        indexed.sort(key=lambda x: x[1].validation_score, reverse=True)
        top_5 = indexed[:5]

        for col, (idx, m) in enumerate(top_5):
            ax = axes[row, col]
            handler.plot_explanation_graph(
                adjs[idx], xs[idx], ax=ax,
                title=f"v={m.validation_score:.2f}"
            )

            if col == 0:
                ax.set_ylabel(model_name.upper(), fontsize=12, fontweight='bold')

    fig.suptitle(f'Best Explanations by k-GNN Model ({handler.name})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")

    return fig


def plot_legend(dataset: str = 'mutag', save_path: str = None):
    """
    Create a legend showing node types and colors for a dataset.

    Args:
        dataset: Dataset name
        save_path: Path to save figure
    """
    handler = get_handler(dataset)
    return handler.plot_legend(save_path=save_path)


# Backward compatibility aliases
def plot_atom_legend(save_path: str = None):
    """Create a legend showing atom types and colors (MUTAG)."""
    return plot_legend('mutag', save_path)


if __name__ == "__main__":
    # Test visualization with all datasets
    print("Testing visualization utilities...")

    os.makedirs('./test_viz', exist_ok=True)

    for dataset_name in ['mutag', 'proteins', 'dd']:
        print(f"\nTesting {dataset_name}...")
        handler = get_handler(dataset_name)

        # Generate random test data appropriate for each dataset
        num_samples = 6
        if dataset_name == 'mutag':
            max_nodes = 15
            num_features = 7
        elif dataset_name == 'proteins':
            max_nodes = 20
            num_features = 3
        else:  # dd
            max_nodes = 25
            num_features = 89

        # Random adjacency (symmetric)
        adjs = np.random.rand(num_samples, max_nodes, max_nodes)
        adjs = (adjs + adjs.transpose(0, 2, 1)) / 2
        adjs = (adjs > 0.8).astype(np.float32)

        # Random one-hot features
        xs = np.zeros((num_samples, max_nodes, num_features))
        for i in range(num_samples):
            for j in range(max_nodes):
                xs[i, j, np.random.randint(min(num_features, 10))] = 1

        # Test grid plot
        plot_explanation_grid(
            adjs, xs,
            save_path=f'./test_viz/grid_{dataset_name}.png',
            dataset=dataset_name
        )

        # Test legend
        plot_legend(dataset_name, save_path=f'./test_viz/legend_{dataset_name}.png')

    print("\nTest visualizations saved to ./test_viz/")
