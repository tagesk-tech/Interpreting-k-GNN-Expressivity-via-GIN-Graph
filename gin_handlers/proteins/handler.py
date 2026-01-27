"""
gin_handlers/proteins/handler.py
PROTEINS-specific GIN handler for protein structure graphs.

PROTEINS Dataset:
- ~1113 protein structures
- Binary classification: Enzyme (1) vs Non-Enzyme (0)
- 3 node features representing amino acid/secondary structure properties
- Graphs represent protein contact maps or structural relationships
"""

from typing import Dict
import matplotlib.pyplot as plt
from ..base import DatasetHandler


class ProteinsHandler(DatasetHandler):
    """Handler for PROTEINS dataset."""

    @property
    def name(self) -> str:
        return "PROTEINS"

    @property
    def node_labels(self) -> Dict[int, str]:
        """
        Node feature mapping for PROTEINS.

        PROTEINS has 3 node features representing secondary structure
        or amino acid properties. The exact meaning depends on the
        specific dataset version, but commonly represents:
        - Secondary structure type (Helix, Sheet, Coil/Turn)
        - Or other biochemical attributes
        """
        return {
            0: 'H',   # Helix (alpha-helix)
            1: 'S',   # Sheet (beta-sheet)
            2: 'C'    # Coil/Turn (loop regions)
        }

    @property
    def node_colors(self) -> Dict[str, str]:
        """
        Colors for secondary structure types.
        Based on standard protein visualization conventions.
        """
        return {
            'H': '#FF6B6B',    # Red/Pink for Helix (alpha-helix)
            'S': '#4ECDC4',    # Cyan/Teal for Sheet (beta-sheet)
            'C': '#95E1D3',    # Light green for Coil/Turn
            '?': '#808080'     # Gray (unknown)
        }

    @property
    def class_names(self) -> Dict[int, str]:
        """Class names for PROTEINS."""
        return {
            0: 'Non-Enzyme',
            1: 'Enzyme'
        }

    def plot_legend(self, save_path: str = None):
        """Create a legend showing secondary structure types for PROTEINS."""
        fig, ax = plt.subplots(figsize=(8, 2))

        structures = list(self.node_labels.values())
        colors = [self.node_colors[s] for s in structures]
        full_names = ['Helix', 'Sheet', 'Coil/Turn']

        for i, (struct, color, full_name) in enumerate(zip(structures, colors, full_names)):
            circle = plt.Circle((i * 1.5 + 0.75, 0.5), 0.35, color=color, ec='black')
            ax.add_patch(circle)
            ax.text(i * 1.5 + 0.75, 0.5, struct, ha='center', va='center',
                    fontsize=14, fontweight='bold')
            ax.text(i * 1.5 + 0.75, -0.1, full_name, ha='center', va='top',
                    fontsize=10)

        ax.set_xlim(0, len(structures) * 1.5)
        ax.set_ylim(-0.4, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Secondary Structure Types in PROTEINS', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Legend saved to: {save_path}")

        return fig
