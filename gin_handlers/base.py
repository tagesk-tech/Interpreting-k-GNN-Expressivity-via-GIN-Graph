"""
gin_handlers/base.py
Base class and factory for dataset-specific GIN handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class DatasetHandler(ABC):
    """
    Abstract base class for dataset-specific GIN handlers.

    Each dataset handler provides:
    - Node feature labels and colors for visualization
    - Class names for the classification task
    - Domain-specific visualization methods
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass

    @property
    @abstractmethod
    def node_labels(self) -> Dict[int, str]:
        """Mapping from node feature index to label."""
        pass

    @property
    @abstractmethod
    def node_colors(self) -> Dict[str, str]:
        """Mapping from node label to hex color."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Mapping from class index to class name."""
        pass

    @property
    def unknown_color(self) -> str:
        """Color for unknown/unlabeled nodes."""
        return '#808080'  # Gray

    def get_node_label(self, feature_idx: int) -> str:
        """Get label for a node feature index."""
        return self.node_labels.get(feature_idx, '?')

    def get_node_color(self, label: str) -> str:
        """Get color for a node label."""
        return self.node_colors.get(label, self.unknown_color)

    def get_class_name(self, class_idx: int) -> str:
        """Get name for a class index."""
        return self.class_names.get(class_idx, f'Class {class_idx}')

    def adj_to_networkx(
        self,
        adj: np.ndarray,
        x: np.ndarray,
        edge_threshold: float = 0.5,
        remove_isolated: bool = True
    ) -> Tuple[nx.Graph, Dict[int, str], List[str]]:
        """
        Convert adjacency matrix and features to NetworkX graph.

        Args:
            adj: Adjacency matrix [N, N]
            x: Node features [N, D] (one-hot or multi-dimensional)
            edge_threshold: Threshold for edge existence
            remove_isolated: Whether to remove isolated nodes

        Returns:
            G: NetworkX graph
            node_labels: Dict mapping node index to label
            node_colors: List of colors for each node
        """
        # Create graph from edges above threshold
        edges = (adj > edge_threshold).astype(np.float32)
        np.fill_diagonal(edges, 0)  # Remove self-loops

        rows, cols = np.where(edges > 0)
        edge_list = [(int(r), int(c)) for r, c in zip(rows, cols) if r < c]

        G = nx.Graph()
        G.add_edges_from(edge_list)

        if remove_isolated:
            G.remove_nodes_from(list(nx.isolates(G)))

        # Get feature indices (argmax for one-hot, or primary feature)
        if x.ndim == 2:
            feature_indices = np.argmax(x, axis=1)
        else:
            feature_indices = x.astype(int)

        # Create labels and colors for remaining nodes
        node_labels_dict = {}
        colors_list = []

        for node_idx in G.nodes():
            feat_idx = feature_indices[node_idx]
            label = self.get_node_label(feat_idx)
            node_labels_dict[node_idx] = label
            colors_list.append(self.get_node_color(label))

        return G, node_labels_dict, colors_list

    def plot_explanation_graph(
        self,
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

        G, node_labels, node_colors = self.adj_to_networkx(adj, x, edge_threshold)

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
        self,
        adjs: np.ndarray,
        xs: np.ndarray,
        metrics: List = None,
        num_cols: int = 5,
        figsize: Tuple[int, int] = None,
        save_path: str = None,
        title: str = None
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
        if title is None:
            title = f"Generated Explanation Graphs ({self.name})"

        num_graphs = adjs.shape[0]
        num_rows = (num_graphs + num_cols - 1) // num_cols

        if figsize is None:
            figsize = (3 * num_cols, 3 * num_rows)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_graphs):
            row, col = i // num_cols, i % num_cols
            ax = axes[row, col]

            if metrics is not None:
                m = metrics[i]
                subtitle = f"v={m.validation_score:.2f}, n={m.num_nodes}"
            else:
                subtitle = f"Sample {i+1}"

            self.plot_explanation_graph(adjs[i], xs[i], ax=ax, title=subtitle)

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

    @abstractmethod
    def plot_legend(self, save_path: str = None):
        """Plot a legend showing node types and colors."""
        pass


def get_handler(dataset_name: str) -> DatasetHandler:
    """
    Factory function to get the appropriate handler for a dataset.

    Args:
        dataset_name: Name of the dataset (mutag, proteins, dd)

    Returns:
        DatasetHandler instance for the dataset
    """
    name_lower = dataset_name.lower()

    if name_lower == 'mutag':
        from .mutag import MutagHandler
        return MutagHandler()
    elif name_lower == 'proteins':
        from .proteins import ProteinsHandler
        return ProteinsHandler()
    elif name_lower == 'dd':
        from .dd import DDHandler
        return DDHandler()
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available handlers: mutag, proteins, dd"
        )
