#!/usr/bin/env python
"""
visualize_standalone.py
Standalone visualization of GIN-Graph .npz sample files.

Strictly decoupled from the training pipeline — only requires
numpy, matplotlib, networkx, and argparse. No torch or local model imports.

Usage:
    python visualize_standalone.py results/samples_mutag_12gnn_epoch100.npz
    python visualize_standalone.py results/samples_mutag_12gnn_epoch100.npz --output fig.png
    python visualize_standalone.py results/samples_mutag_12gnn_epoch100.npz --threshold 0.3
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# Hardcoded dataset labels (auto-detected from feature dim D)
# ============================================================

MUTAG_LABELS = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
MUTAG_COLORS = {
    'C': '#FFA500', 'N': '#00BFFF', 'O': '#FF0000', 'F': '#32CD32',
    'I': '#800080', 'Cl': '#90EE90', 'Br': '#8B4513', '?': '#808080',
}

PROTEINS_LABELS = {0: 'Helix', 1: 'Sheet', 2: 'Turn'}
PROTEINS_COLORS = {
    'Helix': '#E74C3C', 'Sheet': '#3498DB', 'Turn': '#2ECC71', '?': '#808080',
}


def detect_dataset(D):
    """Auto-detect dataset from feature dimension and return (name, labels, colors)."""
    if D == 7:
        return 'MUTAG', MUTAG_LABELS, MUTAG_COLORS
    elif D == 3:
        return 'PROTEINS', PROTEINS_LABELS, PROTEINS_COLORS
    else:
        # Generic fallback (DD has D=89, or any unknown dataset)
        cmap = plt.cm.get_cmap('tab20')
        labels = {i: f'F{i}' for i in range(D)}
        colors = {f'F{i}': cmap(i % 20 / 20.0) for i in range(D)}
        colors['?'] = '#808080'
        return f'D={D}', labels, colors


def build_graph(adj, x, labels, threshold=0.5):
    """
    Build a networkx Graph from a single adjacency matrix and feature vector.

    Thresholds adj at the given value, assigns node labels from argmax(x),
    and removes isolated nodes for cleaner visualization.
    """
    N = adj.shape[0]
    G = nx.Graph()

    node_types = np.argmax(x, axis=-1)

    for i in range(N):
        G.add_node(i, label=labels.get(int(node_types[i]), '?'))

    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] > threshold:
                G.add_edge(i, j)

    # Remove isolated nodes (degree 0)
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def plot_graph(G, ax, colors, title=''):
    """Draw a single graph onto a matplotlib Axes."""
    if len(G) == 0:
        ax.text(0.5, 0.5, 'Empty graph', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#999999')
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        return

    node_colors = [colors.get(G.nodes[n]['label'], colors.get('?', '#808080'))
                   for n in G.nodes]
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes}

    pos = nx.spring_layout(G, seed=42, k=1.5 / max(len(G) ** 0.5, 1))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#888888', width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=250, edgecolors='#333333', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=6)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GIN-Graph .npz sample files (standalone, no torch needed)')
    parser.add_argument('npz_path', type=str, help='Path to .npz sample file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save figure to file instead of showing (e.g. fig.png)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Edge threshold for adjacency (default: 0.5)')
    parser.add_argument('--cols', type=int, default=4,
                        help='Number of columns in the grid (default: 4)')
    args = parser.parse_args()

    # Load data
    data = np.load(args.npz_path)
    adjs = data['adjs']
    xs = data['xs']
    predictions = data['predictions'] if 'predictions' in data else None
    epoch = int(data['epoch']) if 'epoch' in data else None

    B, N, D = xs.shape
    dataset_name, labels, colors = detect_dataset(D)

    # Grid layout (up to 4 rows x cols columns = 16 by default)
    num_plots = min(B, 4 * args.cols)
    ncols = args.cols
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    # Normalize axes to 2D array
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    epoch_str = f'  —  Epoch {epoch}' if epoch is not None else ''
    fig.suptitle(f'{dataset_name} Generated Samples{epoch_str}',
                 fontsize=13, fontweight='bold', y=1.0)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        if idx < num_plots:
            G = build_graph(adjs[idx], xs[idx], labels, threshold=args.threshold)
            parts = [f'#{idx}', f'{len(G)}n', f'{G.number_of_edges()}e']
            if predictions is not None:
                parts.append(f'p={predictions[idx]:.2f}')
            plot_graph(G, ax, colors, title=' | '.join(parts))
        else:
            ax.axis('off')

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f'Saved to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
