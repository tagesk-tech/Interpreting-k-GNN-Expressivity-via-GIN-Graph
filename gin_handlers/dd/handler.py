"""
gin_handlers/dd/handler.py
DD-specific GIN handler for protein structure graphs.

DD (D&D) Dataset:
- ~1178 protein structures
- Binary classification: Enzyme (1) vs Non-Enzyme (0)
- 89 node features representing amino acid types and properties
- Larger graphs than MUTAG/PROTEINS (up to 500 nodes)
"""

from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ..base import DatasetHandler


# Amino acid one-letter codes (standard 20)
AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]

# Amino acid property-based coloring
AMINO_ACID_COLORS = {
    # Hydrophobic (yellow/orange)
    'A': '#FFD700', 'V': '#FFA500', 'I': '#FF8C00', 'L': '#FF7F50',
    'M': '#FF6347', 'F': '#FFB6C1', 'W': '#FF69B4', 'P': '#FFDAB9',
    # Polar (green)
    'S': '#90EE90', 'T': '#98FB98', 'N': '#00FA9A', 'Q': '#00FF7F',
    'Y': '#32CD32', 'C': '#7CFC00',
    # Charged positive (blue)
    'K': '#4169E1', 'R': '#0000FF', 'H': '#6495ED',
    # Charged negative (red)
    'D': '#DC143C', 'E': '#FF0000',
    # Special
    'G': '#D3D3D3',  # Glycine (gray - flexible)
    '?': '#808080'   # Unknown
}


class DDHandler(DatasetHandler):
    """Handler for DD (D&D) protein dataset."""

    @property
    def name(self) -> str:
        return "DD"

    @property
    def node_labels(self) -> Dict[int, str]:
        """
        Node feature mapping for DD.

        DD has 89 node features. The first 20 typically represent
        amino acid types, with additional features for structural
        or biochemical properties. We use simplified labeling.
        """
        labels = {}
        # First 20 are amino acids
        for i, aa in enumerate(AMINO_ACIDS):
            labels[i] = aa
        # Remaining features get generic labels
        for i in range(20, 89):
            labels[i] = f'F{i}'  # Feature i
        return labels

    @property
    def node_colors(self) -> Dict[str, str]:
        """Colors based on amino acid properties."""
        colors = AMINO_ACID_COLORS.copy()
        # Add colors for generic features
        for i in range(20, 89):
            colors[f'F{i}'] = '#A9A9A9'  # Dark gray for extended features
        return colors

    @property
    def class_names(self) -> Dict[int, str]:
        """Class names for DD."""
        return {
            0: 'Non-Enzyme',
            1: 'Enzyme'
        }

    def plot_legend(self, save_path: str = None):
        """Create a legend showing amino acid types grouped by property."""
        fig, ax = plt.subplots(figsize=(12, 4))

        # Group amino acids by property
        groups = {
            'Hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'],
            'Polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
            'Positive': ['K', 'R', 'H'],
            'Negative': ['D', 'E'],
            'Special': ['G']
        }

        y_offset = 0.8
        for group_name, aas in groups.items():
            ax.text(0, y_offset, group_name + ':', fontsize=10, fontweight='bold')
            for i, aa in enumerate(aas):
                color = AMINO_ACID_COLORS.get(aa, '#808080')
                circle = plt.Circle((1.5 + i * 0.6, y_offset), 0.15,
                                    color=color, ec='black')
                ax.add_patch(circle)
                ax.text(1.5 + i * 0.6, y_offset, aa, ha='center', va='center',
                        fontsize=8, fontweight='bold')
            y_offset -= 0.25

        ax.set_xlim(-0.5, 8)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Amino Acid Types in DD (grouped by property)', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Legend saved to: {save_path}")

        return fig
