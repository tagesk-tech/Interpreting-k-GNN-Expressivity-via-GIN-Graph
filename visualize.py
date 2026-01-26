"""
visualize.py
Visualization utilities for k-GNN and GIN-Graph results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional, Dict, Tuple
import os

from config import ATOM_LABELS, ATOM_COLORS, CLASS_NAMES


def adj_to_networkx(
    adj: np.ndarray,
    x: np.ndarray,
    edge_threshold: float = 0.5,
    remove_isolated: bool = True
) -> Tuple[nx.Graph, Dict[int, str], List[str]]:
    """
    Convert adjacency matrix and features to NetworkX graph.
    
    Args:
        adj: Adjacency matrix [N, N]
        x: Node features [N, D] (one-hot)
        edge_threshold: Threshold for edge existence
        remove_isolated: Whether to remove isolated nodes
        
    Returns:
        G: NetworkX graph
        node_labels: Dict mapping node index to atom label
        node_colors: List of colors for each node
    """
    # Create graph from edges above threshold
    edges = (adj > edge_threshold).astype(np.float32)
    np.fill_diagonal(edges, 0)  # Remove self-loops
    
    rows, cols = np.where(edges > 0)
    edge_list = [(int(r), int(c)) for r, c in zip(rows, cols) if r < c]  # Upper triangle only
    
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Remove isolated nodes if requested
    if remove_isolated:
        G.remove_nodes_from(list(nx.isolates(G)))
    
    # Get atom types
    atom_indices = np.argmax(x, axis=1)
    
    # Create labels and colors for remaining nodes
    node_labels = {}
    node_colors = []
    
    for node_idx in G.nodes():
        atom_type = atom_indices[node_idx]
        label = ATOM_LABELS.get(atom_type, '?')
        node_labels[node_idx] = label
        node_colors.append(ATOM_COLORS.get(label, ATOM_COLORS['?']))
    
    return G, node_labels, node_colors


def plot_explanation_graph(
    adj: np.ndarray,
    x: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    edge_threshold: float = 0.5,
    node_size: int = 400,
    font_size: int = 10,
    seed: int = 42
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
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    G, node_labels, node_colors = adj_to_networkx(adj, x, edge_threshold)
    
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=seed)
        nx.draw(
            G, pos, ax=ax,
            with_labels=True,
            labels=node_labels,
            node_color=node_colors,
            node_size=node_size,
            edge_color='black',
            font_size=font_size,
            font_weight='bold'
        )
    else:
        ax.text(0.5, 0.5, "Empty Graph", ha='center', va='center', fontsize=12)
    
    ax.set_title(title)
    ax.axis('off')


def plot_explanation_grid(
    adjs: np.ndarray,
    xs: np.ndarray,
    metrics: List = None,
    num_cols: int = 5,
    figsize: Tuple[int, int] = None,
    save_path: str = None,
    title: str = "Generated Explanation Graphs"
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
    """
    num_graphs = adjs.shape[0]
    num_rows = (num_graphs + num_cols - 1) // num_cols
    
    if figsize is None:
        figsize = (3 * num_cols, 3 * num_rows)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_graphs):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col]
        
        if metrics is not None:
            m = metrics[i]
            subtitle = f"v={m.validation_score:.2f}, n={m.num_nodes}"
        else:
            subtitle = f"Sample {i+1}"
        
        plot_explanation_graph(adjs[i], xs[i], ax=ax, title=subtitle)
    
    # Hide empty subplots
    for i in range(num_graphs, num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        axes[row, col].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


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
    ax.set_ylabel('Lambda (λ)')
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
    ax.axvline(x=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    ax.set_xlabel('Validation Score')
    ax.set_ylabel('Count')
    ax.set_title('Validation Score Distribution')
    ax.legend()
    
    # Prediction probabilities
    ax = axes[0, 1]
    ax.hist(probs, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(x=np.mean(probs), color='r', linestyle='--', label=f'Mean: {np.mean(probs):.3f}')
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    
    # Degree scores
    ax = axes[1, 0]
    ax.hist(degrees, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=np.mean(degrees), color='r', linestyle='--', label=f'Mean: {np.mean(degrees):.3f}')
    ax.set_xlabel('Degree Score')
    ax.set_ylabel('Count')
    ax.set_title('Degree Score Distribution')
    ax.legend()
    
    # Valid vs Invalid
    ax = axes[1, 1]
    valid_count = sum(valid)
    invalid_count = len(valid) - valid_count
    ax.bar(['Valid', 'Invalid'], [valid_count, invalid_count], color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Validity: {valid_count}/{len(valid)} ({100*valid_count/len(valid):.1f}%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics distribution saved to: {save_path}")
    
    return fig


def create_comparison_figure(
    results: Dict[str, Dict],
    save_path: str = None
):
    """
    Create comparison figure across different k-GNN models.
    
    Args:
        results: Dict of model_name -> {adjs, xs, metrics, ...}
        save_path: Path to save figure
    """
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
            plot_explanation_graph(adjs[idx], xs[idx], ax=ax, 
                                   title=f"v={m.validation_score:.2f}")
            
            if col == 0:
                ax.set_ylabel(model_name.upper(), fontsize=12, fontweight='bold')
    
    fig.suptitle('Best Explanations by k-GNN Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    
    return fig


def plot_atom_legend(save_path: str = None):
    """Create a legend showing atom types and colors."""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    atoms = list(ATOM_LABELS.values())
    colors = [ATOM_COLORS[a] for a in atoms]
    
    for i, (atom, color) in enumerate(zip(atoms, colors)):
        circle = plt.Circle((i + 0.5, 0.5), 0.3, color=color, ec='black')
        ax.add_patch(circle)
        ax.text(i + 0.5, 0.5, atom, ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, len(atoms))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Atom Types in MUTAG', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test visualization with random data
    print("Testing visualization utilities...")
    
    # Generate random test data
    num_samples = 10
    max_nodes = 15
    num_features = 7
    
    # Random adjacency (symmetric)
    adjs = np.random.rand(num_samples, max_nodes, max_nodes)
    adjs = (adjs + adjs.transpose(0, 2, 1)) / 2
    adjs = (adjs > 0.7).astype(np.float32)
    
    # Random one-hot features
    xs = np.zeros((num_samples, max_nodes, num_features))
    for i in range(num_samples):
        for j in range(max_nodes):
            xs[i, j, np.random.randint(num_features)] = 1
    
    # Create test output directory
    os.makedirs('./test_viz', exist_ok=True)
    
    # Test grid plot
    plot_explanation_grid(adjs[:6], xs[:6], save_path='./test_viz/grid.png')
    
    # Test atom legend
    plot_atom_legend(save_path='./test_viz/legend.png')
    
    print("Test visualizations saved to ./test_viz/")
